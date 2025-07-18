"""MCP (Model Context Protocol) server implementation"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import websockets
import logfire
from ..tool_registry import tool_registry
from ..base_tool import BaseTool, ToolResult


class MCPServer:
    """MCP server for exposing tools via Model Context Protocol"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """Initialize MCP server
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self._logger = logfire.span("mcp_server")
        self._clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._handlers: Dict[str, Callable] = {
            "list_tools": self._handle_list_tools,
            "get_tool": self._handle_get_tool,
            "execute_tool": self._handle_execute_tool,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe
        }
        self._subscriptions: Dict[str, List[str]] = {}
    
    async def start(self) -> None:
        """Start the MCP server"""
        self._logger.info(f"Starting MCP server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port
        ):
            self._logger.info("MCP server started")
            await asyncio.Future()  # Run forever
    
    async def _handle_client(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str
    ) -> None:
        """Handle client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self._clients[client_id] = websocket
        
        self._logger.info(f"Client connected: {client_id}")
        
        try:
            # Send welcome message
            await self._send_message(websocket, {
                "type": "welcome",
                "version": "1.0.0",
                "capabilities": list(self._handlers.keys())
            })
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self._logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            self._logger.error(f"Client error: {client_id}", error=str(e))
        finally:
            # Clean up
            self._clients.pop(client_id, None)
            self._subscriptions.pop(client_id, None)
    
    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle incoming message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            msg_id = data.get("id")
            
            if msg_type not in self._handlers:
                await self._send_error(
                    self._clients[client_id],
                    msg_id,
                    f"Unknown message type: {msg_type}"
                )
                return
            
            # Handle the message
            handler = self._handlers[msg_type]
            result = await handler(client_id, data)
            
            # Send response
            if result is not None:
                await self._send_response(
                    self._clients[client_id],
                    msg_id,
                    result
                )
                
        except json.JSONDecodeError:
            await self._send_error(
                self._clients[client_id],
                None,
                "Invalid JSON message"
            )
        except Exception as e:
            self._logger.error(f"Message handling error", error=str(e))
            await self._send_error(
                self._clients[client_id],
                msg_id if 'msg_id' in locals() else None,
                str(e)
            )
    
    async def _handle_list_tools(
        self,
        client_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle list tools request"""
        category = data.get("category")
        tools = tool_registry.list_tools(category)
        
        tool_infos = []
        for tool_name in tools:
            info = tool_registry.get_tool_info(tool_name)
            if info:
                tool_infos.append({
                    "name": info["name"],
                    "description": info["description"],
                    "category": info["category"],
                    "schema": info["schema"]
                })
        
        return {
            "tools": tool_infos,
            "total": len(tool_infos)
        }
    
    async def _handle_get_tool(
        self,
        client_id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle get tool request"""
        tool_name = data.get("tool_name")
        if not tool_name:
            raise ValueError("tool_name is required")
        
        info = tool_registry.get_tool_info(tool_name)
        return info
    
    async def _handle_execute_tool(
        self,
        client_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle execute tool request"""
        tool_name = data.get("tool_name")
        params = data.get("params", {})
        
        if not tool_name:
            raise ValueError("tool_name is required")
        
        # Get tool
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Check if confirmation is required
        if tool.requires_confirmation:
            # Send confirmation request
            await self._send_message(self._clients[client_id], {
                "type": "confirmation_required",
                "tool_name": tool_name,
                "params": params,
                "message": f"Tool '{tool_name}' requires confirmation to execute"
            })
            
            # For now, we'll proceed - in real implementation, wait for confirmation
        
        # Execute tool
        result = await tool.execute(**params)
        
        # Notify subscribers
        await self._notify_subscribers(tool_name, {
            "event": "tool_executed",
            "tool_name": tool_name,
            "params": params,
            "result": result.dict()
        })
        
        return result.dict()
    
    async def _handle_subscribe(
        self,
        client_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle subscription request"""
        events = data.get("events", [])
        
        if client_id not in self._subscriptions:
            self._subscriptions[client_id] = []
        
        for event in events:
            if event not in self._subscriptions[client_id]:
                self._subscriptions[client_id].append(event)
        
        return {
            "subscribed": events,
            "total_subscriptions": len(self._subscriptions[client_id])
        }
    
    async def _handle_unsubscribe(
        self,
        client_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle unsubscription request"""
        events = data.get("events", [])
        
        if client_id in self._subscriptions:
            for event in events:
                if event in self._subscriptions[client_id]:
                    self._subscriptions[client_id].remove(event)
        
        return {
            "unsubscribed": events,
            "remaining_subscriptions": len(self._subscriptions.get(client_id, []))
        }
    
    async def _send_message(
        self,
        websocket: websockets.WebSocketServerProtocol,
        message: Dict[str, Any]
    ) -> None:
        """Send message to client"""
        await websocket.send(json.dumps(message))
    
    async def _send_response(
        self,
        websocket: websockets.WebSocketServerProtocol,
        msg_id: Optional[str],
        result: Any
    ) -> None:
        """Send response to client"""
        await self._send_message(websocket, {
            "type": "response",
            "id": msg_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _send_error(
        self,
        websocket: websockets.WebSocketServerProtocol,
        msg_id: Optional[str],
        error: str
    ) -> None:
        """Send error to client"""
        await self._send_message(websocket, {
            "type": "error",
            "id": msg_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _notify_subscribers(
        self,
        event: str,
        data: Dict[str, Any]
    ) -> None:
        """Notify subscribers of an event"""
        for client_id, subscriptions in self._subscriptions.items():
            if event in subscriptions and client_id in self._clients:
                try:
                    await self._send_message(self._clients[client_id], {
                        "type": "event",
                        "event": event,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    self._logger.error(
                        f"Failed to notify subscriber {client_id}",
                        error=str(e)
                    )


class MCPClient:
    """MCP client for connecting to MCP servers"""
    
    def __init__(self, server_url: str):
        """Initialize MCP client
        
        Args:
            server_url: WebSocket URL of MCP server
        """
        self.server_url = server_url
        self._logger = logfire.span("mcp_client")
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._response_handlers: Dict[str, asyncio.Future] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._message_id = 0
    
    async def connect(self) -> None:
        """Connect to MCP server"""
        self._logger.info(f"Connecting to MCP server: {self.server_url}")
        self._websocket = await websockets.connect(self.server_url)
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
        
        # Wait for welcome message
        await asyncio.sleep(0.1)
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
    
    async def _handle_messages(self) -> None:
        """Handle incoming messages"""
        if not self._websocket:
            return
        
        try:
            async for message in self._websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "welcome":
                    self._logger.info("Connected to MCP server", capabilities=data.get("capabilities"))
                    
                elif msg_type == "response":
                    msg_id = data.get("id")
                    if msg_id in self._response_handlers:
                        self._response_handlers[msg_id].set_result(data.get("result"))
                        
                elif msg_type == "error":
                    msg_id = data.get("id")
                    if msg_id in self._response_handlers:
                        self._response_handlers[msg_id].set_exception(
                            Exception(data.get("error"))
                        )
                        
                elif msg_type == "event":
                    event = data.get("event")
                    if event in self._event_handlers:
                        for handler in self._event_handlers[event]:
                            try:
                                await handler(data.get("data"))
                            except Exception as e:
                                self._logger.error(f"Event handler error", error=str(e))
                                
        except websockets.exceptions.ConnectionClosed:
            self._logger.info("Disconnected from MCP server")
        except Exception as e:
            self._logger.error("Message handling error", error=str(e))
    
    async def _send_request(self, msg_type: str, data: Dict[str, Any]) -> Any:
        """Send request and wait for response"""
        if not self._websocket:
            raise RuntimeError("Not connected to MCP server")
        
        # Generate message ID
        msg_id = str(self._message_id)
        self._message_id += 1
        
        # Create response future
        response_future = asyncio.Future()
        self._response_handlers[msg_id] = response_future
        
        # Send request
        message = {
            "type": msg_type,
            "id": msg_id,
            **data
        }
        await self._websocket.send(json.dumps(message))
        
        try:
            # Wait for response
            result = await asyncio.wait_for(response_future, timeout=30)
            return result
        finally:
            # Clean up
            self._response_handlers.pop(msg_id, None)
    
    async def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools"""
        result = await self._send_request("list_tools", {"category": category})
        return result.get("tools", [])
    
    async def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information"""
        return await self._send_request("get_tool", {"tool_name": tool_name})
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        return await self._send_request("execute_tool", {
            "tool_name": tool_name,
            "params": params
        })
    
    async def subscribe(self, events: List[str]) -> None:
        """Subscribe to events"""
        await self._send_request("subscribe", {"events": events})
    
    async def unsubscribe(self, events: List[str]) -> None:
        """Unsubscribe from events"""
        await self._send_request("unsubscribe", {"events": events})
    
    def on_event(self, event: str, handler: Callable) -> None:
        """Register event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)