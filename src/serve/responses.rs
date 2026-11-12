//! Profil Responses texte/fonctions, historique fourni par le client et sans stockage.

use std::io::Write;

use serde_json::{json, Value};

use super::error::{ServeError, ServeResult};
use super::flint_output::FlintEvent;
use super::http::{send_json, write_headers};
use super::protocol::{created_now, ChatCompletionChunk, ToolCall};
use super::state::{ServeState, ServedCompletion};
use super::streaming::{CompletionSink, CompletionStreamEvent};
use super::tool_output::ToolOutputParser;

mod request;
#[cfg(test)]
mod tests;
use request::Request;

pub(super) fn handle<S: Write>(
    stream: &mut S,
    state: &mut ServeState,
    body: &[u8],
) -> ServeResult<()> {
    let request = Request::parse(body, state.max_tokens_cap())?;
    let parser = state.response_output_parser(&request.chat)?;
    let response = Response::new(&request);
    if request.chat.stream {
        let mut sink = ResponsesSse::new(stream, response, parser);
        let result = state.complete_streaming(request.chat, None, |event| sink.event(event));
        sink.finish(result)
    } else {
        let completion = state.complete(request.chat, None)?;
        let mut response = response;
        response.value["output"] = output(&completion.content, parser);
        response.complete(&completion);
        send_json(stream, 200, &response.value, completion.metric_headers())
    }
}

fn identity(prefix: &str) -> String {
    format!("{prefix}_{}", ChatCompletionChunk::stream_id())
}

fn message(text: &str, status: &str) -> Value {
    json!({"id":identity("msg"),"type":"message","role":"assistant","status":status,
        "content":[part(text)]})
}

fn part(text: &str) -> Value {
    json!({"type":"output_text","text":text,"annotations":[],"logprobs":[]})
}

fn function(call: ToolCall) -> Value {
    json!({"id":identity("fc"),"type":"function_call","status":"completed",
        "call_id":call.id,"name":call.function.name,"arguments":call.function.arguments})
}

fn output(content: &str, parser: Option<ToolOutputParser>) -> Value {
    let mut items = Vec::new();
    match parser {
        Some(parser) => {
            let parsed = parser.parse(content);
            if !parsed.content.is_empty() {
                items.push(message(&parsed.content, "completed"));
            }
            items.extend(parsed.tool_calls.into_iter().map(function));
        }
        None => items.push(message(content, "completed")),
    }
    Value::Array(items)
}

struct Response {
    value: Value,
}

impl Response {
    fn new(request: &Request) -> Self {
        Self {
            value: json!({
                "id":identity("resp"),"object":"response","created_at":created_now(),
                "status":"in_progress","model":request.chat.model,"output":[],"error":null,
                "incomplete_details":null,"usage":null,"store":false,"background":false,
                "instructions":request.options.get("instructions"),
                "max_output_tokens":request.chat.max_tokens(),
                "temperature":request.chat.temperature.unwrap_or(0.0),
                "top_p":request.chat.top_p.unwrap_or(1.0),
                "parallel_tool_calls":request.chat.parallel_tool_calls(),
                "tool_choice":request.options.get("tool_choice").filter(|v| !v.is_null()).cloned().unwrap_or(json!("auto")),
                "tools":request.options.get("tools").cloned().unwrap_or(json!([])),
                "text":{"format":{"type":"text"}},"truncation":"disabled",
                "metadata":{},"previous_response_id":null
            }),
        }
    }

    fn complete(&mut self, completion: &ServedCompletion) {
        self.value["model"] = json!(completion.model);
        let incomplete = completion.finish_reason == "length";
        self.value["status"] = json!(if incomplete {
            "incomplete"
        } else {
            "completed"
        });
        if incomplete {
            self.value["incomplete_details"] = json!({"reason":"max_output_tokens"});
            if let Some(items) = self.value["output"].as_array_mut() {
                for item in items.iter_mut().filter(|item| item["type"] == "message") {
                    item["status"] = json!("incomplete");
                }
            }
        }
        let usage = completion.usage;
        self.value["usage"] = json!({
            "input_tokens":usage.prompt_tokens(),
            "input_tokens_details":{"cached_tokens":completion.reused_prefix_tokens,"cache_write_tokens":0},
            "output_tokens":usage.completion_tokens(),
            "output_tokens_details":{"reasoning_tokens":0},
            "total_tokens":usage.prompt_tokens() + usage.completion_tokens()
        });
    }
}

/// Les appels sont publiés après validation complète par le parseur commun : les
/// arguments arrivent en un delta JSON, sans exposer de fragment invalide.
struct ResponsesSse<'a, S> {
    stream: &'a mut S,
    response: Response,
    parser: Option<ToolOutputParser>,
    started: bool,
    terminal: bool,
    sequence: u64,
    items: Vec<Value>,
    text_index: Option<usize>,
}

impl<'a, S: Write> ResponsesSse<'a, S> {
    fn new(stream: &'a mut S, response: Response, parser: Option<ToolOutputParser>) -> Self {
        Self {
            stream,
            response,
            parser,
            started: false,
            terminal: false,
            sequence: 0,
            items: Vec::new(),
            text_index: None,
        }
    }

    fn emit(&mut self, kind: &str, mut event: Value) -> ServeResult<()> {
        event["type"] = json!(kind);
        event["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        let data = serde_json::to_string(&event)
            .map_err(|e| ServeError::json("événement Responses", e))?;
        write!(self.stream, "event: {kind}\ndata: {data}\n\n")
            .map_err(|e| ServeError::io("SSE Responses", e))?;
        self.stream
            .flush()
            .map_err(|e| ServeError::io("flush Responses", e))
    }

    fn text(&mut self, delta: &str) -> ServeResult<()> {
        let index = match self.text_index {
            Some(index) => index,
            None => {
                let index = self.items.len();
                let mut item = message("", "in_progress");
                item["content"] = json!([]);
                self.emit(
                    "response.output_item.added",
                    json!({"output_index":index,"item":item}),
                )?;
                item["content"] = json!([part("")]);
                self.emit(
                    "response.content_part.added",
                    json!({"item_id":item["id"],"output_index":index,
                    "content_index":0,"part":part("")}),
                )?;
                self.items.push(item);
                self.text_index = Some(index);
                index
            }
        };
        if !delta.is_empty() {
            self.emit(
                "response.output_text.delta",
                json!({"item_id":self.items[index]["id"],
                "output_index":index,"content_index":0,"delta":delta,"logprobs":[]}),
            )?;
            if let Value::String(text) = &mut self.items[index]["content"][0]["text"] {
                text.push_str(delta);
            }
        }
        Ok(())
    }

    fn close_text(&mut self, status: &str) -> ServeResult<()> {
        let Some(index) = self.text_index.take() else {
            return Ok(());
        };
        self.items[index]["status"] = json!(status);
        let item = self.items[index].clone();
        self.emit(
            "response.output_text.done",
            json!({"item_id":item["id"],"output_index":index,
            "content_index":0,"text":item["content"][0]["text"],"logprobs":[]}),
        )?;
        self.emit(
            "response.content_part.done",
            json!({"item_id":item["id"],"output_index":index,
            "content_index":0,"part":item["content"][0]}),
        )?;
        self.emit(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        )
    }

    fn tool_events(&mut self, events: Vec<FlintEvent>) -> ServeResult<()> {
        for event in events {
            match event {
                // Le profil ne fournit pas de sortie reasoning ni de reprise de celle-ci.
                FlintEvent::Reasoning(_) => {}
                FlintEvent::Content(delta) => self.text(&delta)?,
                FlintEvent::ToolCall(call) => {
                    self.close_text("completed")?;
                    let index = self.items.len();
                    let item = function(call);
                    let mut added = item.clone();
                    added["arguments"] = json!("");
                    added["status"] = json!("in_progress");
                    self.emit(
                        "response.output_item.added",
                        json!({"output_index":index,"item":added}),
                    )?;
                    self.emit(
                        "response.function_call_arguments.delta",
                        json!({"item_id":item["id"],
                        "output_index":index,"delta":item["arguments"]}),
                    )?;
                    self.emit(
                        "response.function_call_arguments.done",
                        json!({"item_id":item["id"],
                        "output_index":index,"arguments":item["arguments"],"name":item["name"]}),
                    )?;
                    self.emit(
                        "response.output_item.done",
                        json!({"output_index":index,"item":item}),
                    )?;
                    self.items.push(item);
                }
            }
        }
        Ok(())
    }

    fn fail(&mut self, message: &str) -> ServeResult<()> {
        if self.terminal {
            return Ok(());
        }
        self.terminal = true;
        self.response.value["status"] = json!("failed");
        self.response.value["error"] = json!({"code":"server_error","message":message});
        self.response.value["output"] = json!(self.items);
        self.emit("response.failed", json!({"response":self.response.value}))
    }
}

impl<S: Write> CompletionSink for ResponsesSse<'_, S> {
    fn event(&mut self, event: CompletionStreamEvent<'_>) -> ServeResult<()> {
        if self.terminal {
            return Ok(());
        }
        match event {
            CompletionStreamEvent::Start(start) => {
                self.started = true;
                self.response.value["model"] = json!(start.model);
                write_headers(
                    self.stream,
                    200,
                    "OK",
                    "text/event-stream; charset=utf-8",
                    None,
                    start.metric_headers(),
                )?;
                self.emit("response.created", json!({"response":self.response.value}))?;
                self.emit(
                    "response.in_progress",
                    json!({"response":self.response.value}),
                )
            }
            CompletionStreamEvent::Delta(delta) => match self.parser.as_mut() {
                Some(parser) => {
                    let events = parser.push(delta);
                    self.tool_events(events)
                }
                None => self.text(delta),
            },
            CompletionStreamEvent::TerminalError(error) => self.fail(error.message),
        }
    }

    fn finish(&mut self, result: ServeResult<ServedCompletion>) -> ServeResult<()> {
        if self.terminal {
            return Ok(());
        }
        let completion = match result {
            Ok(completion) => completion,
            Err(error) if self.started => return self.fail(&error.to_string()),
            Err(error) => return Err(error),
        };
        if let Some(parser) = self.parser.as_mut() {
            let events = parser.finish();
            self.tool_events(events)?;
        } else if self.items.is_empty() {
            self.text("")?;
        }
        self.close_text(if completion.finish_reason == "length" {
            "incomplete"
        } else {
            "completed"
        })?;
        self.response.value["output"] = json!(self.items);
        self.response.complete(&completion);
        self.terminal = true;
        let kind = if completion.finish_reason == "length" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        self.emit(kind, json!({"response":self.response.value}))
    }
}
