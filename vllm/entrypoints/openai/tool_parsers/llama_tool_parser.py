import json
import re
from typing import Dict, List, Sequence, Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)

class Llama31ToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

        self.tool_call_start_token = "<|python_tag|>"
        self.tool_call_end_token = "<|eom_id|>"
        self.json_tool_call_regex = re.compile(r'({[^}]+})')

        self.tool_call_start_token_id = self.model_tokenizer.encode(self.tool_call_start_token)[0]
        self.tool_call_end_token_id = self.model_tokenizer.encode(self.tool_call_end_token)[0]

    def extract_tool_calls(self, model_output: str) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

        tool_calls = []
        content = model_output.split(self.tool_call_start_token)[0].strip()

        try:
            tool_call_sections = model_output.split(self.tool_call_start_token)[1:]
            for section in tool_call_sections:
                tool_call_text = section.split(self.tool_call_end_token)[0].strip()
                
                if '.' in tool_call_text:  # Built-in tool call
                    function_name, args_str = tool_call_text.split('.call(')
                    args_str = args_str.rstrip(')')
                    arguments = dict(arg.split('=') for arg in args_str.split(', '))
                    tool_calls.append(ToolCall(
                        type="function",
                        function=FunctionCall(name=function_name, arguments=json.dumps(arguments))
                    ))
                else:  # Custom JSON tool call
                    json_match = self.json_tool_call_regex.search(tool_call_text)
                    if json_match:
                        tool_call_dict = json.loads(json_match.group(1))
                        tool_calls.append(ToolCall(
                            type="function",
                            function=FunctionCall(name=tool_call_dict['name'], arguments=json.dumps(tool_call_dict['parameters']))
                        ))

            return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls, content=content)
        except Exception as e:
            logger.error(f"Error in extracting tool call from response: {e}")
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        if self.tool_call_start_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        try:
            if self.tool_call_start_token_id in delta_token_ids:
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.current_tool_initial_sent = False
                self.streamed_args_for_tool.append("")
                return None

            if not self.current_tool_initial_sent:
                self.current_tool_initial_sent = True
                return DeltaMessage(tool_calls=[InitialDeltaToolCall(index=self.current_tool_id).model_dump(exclude_none=True)])

            tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
            
            if '.' in tool_call_portion:  # Built-in tool call
                function_name, args_portion = tool_call_portion.split('.call(', 1)
                if not self.current_tool_name_sent:
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(name=function_name).model_dump(exclude_none=True))])
                
                args_str = args_portion.split(self.tool_call_end_token)[0].rstrip(')')
                new_args = dict(arg.split('=') for arg in args_str.split(', ') if '=' in arg)
                
                if self.prev_tool_call_arr:
                    prev_args = self.prev_tool_call_arr[-1].get('arguments', {})
                    args_diff = {k: v for k, v in new_args.items() if k not in prev_args or prev_args[k] != v}
                else:
                    args_diff = new_args
                
                if args_diff:
                    self.prev_tool_call_arr.append({'name': function_name, 'arguments': new_args})
                    return DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=json.dumps(args_diff)).model_dump(exclude_none=True))])
            
            else:  # Custom JSON tool call
                json_match = self.json_tool_call_regex.search(tool_call_portion)
                if json_match:
                    tool_call_dict = json.loads(json_match.group(1))
                    if not self.current_tool_name_sent:
                        self.current_tool_name_sent = True
                        return DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(name=tool_call_dict['name']).model_dump(exclude_none=True))])
                    
                    new_args = tool_call_dict['parameters']
                    if self.prev_tool_call_arr:
                        prev_args = self.prev_tool_call_arr[-1].get('parameters', {})
                        args_diff = {k: v for k, v in new_args.items() if k not in prev_args or prev_args[k] != v}
                    else:
                        args_diff = new_args
                    
                    if args_diff:
                        self.prev_tool_call_arr.append(tool_call_dict)
                        return DeltaMessage(tool_calls=[DeltaToolCall(index=self.current_tool_id, function=DeltaFunctionCall(arguments=json.dumps(args_diff)).model_dump(exclude_none=True))])

            return None

        except Exception as e:
            logger.error(f"Error trying to handle streaming tool call: {e}")
            return None
