"""File operation tools"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
import os
import shutil
from ..base_tool import BaseTool, ToolParameter, ToolResult


class ReadFileTool(BaseTool):
    """Tool for reading file contents"""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read contents of a file"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to read",
                required=True
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute file read operation"""
        file_path = Path(params["file_path"])
        encoding = params.get("encoding", "utf-8")
        
        if not file_path.exists():
            return ToolResult(
                success=False,
                error=f"File not found: {file_path}"
            )
        
        if not file_path.is_file():
            return ToolResult(
                success=False,
                error=f"Path is not a file: {file_path}"
            )
        
        try:
            async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                content = await f.read()
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": str(file_path),
                    "size": file_path.stat().st_size,
                    "encoding": encoding
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}"
            )


class WriteFileTool(BaseTool):
    """Tool for writing content to a file"""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def requires_confirmation(self) -> bool:
        return True
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file to write",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            ),
            ToolParameter(
                name="create_dirs",
                type="boolean",
                description="Create parent directories if they don't exist",
                required=False,
                default=True
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="Overwrite existing file",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute file write operation"""
        file_path = Path(params["file_path"])
        content = params["content"]
        encoding = params.get("encoding", "utf-8")
        create_dirs = params.get("create_dirs", True)
        overwrite = params.get("overwrite", False)
        
        # Check if file exists and overwrite is False
        if file_path.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=f"File already exists: {file_path}. Set overwrite=True to overwrite."
            )
        
        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.parent.exists():
            return ToolResult(
                success=False,
                error=f"Parent directory does not exist: {file_path.parent}"
            )
        
        try:
            async with aiofiles.open(file_path, mode='w', encoding=encoding) as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                data={
                    "file_path": str(file_path),
                    "bytes_written": len(content.encode(encoding)),
                    "encoding": encoding
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {str(e)}"
            )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents"""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List contents of a directory"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="directory_path",
                type="string",
                description="Path to the directory",
                required=True
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="File pattern to match (e.g., '*.py')",
                required=False,
                default="*"
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="List recursively",
                required=False,
                default=False
            ),
            ToolParameter(
                name="include_hidden",
                type="boolean",
                description="Include hidden files",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute directory listing"""
        dir_path = Path(params["directory_path"])
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)
        
        if not dir_path.exists():
            return ToolResult(
                success=False,
                error=f"Directory not found: {dir_path}"
            )
        
        if not dir_path.is_dir():
            return ToolResult(
                success=False,
                error=f"Path is not a directory: {dir_path}"
            )
        
        try:
            items = []
            
            if recursive:
                paths = dir_path.rglob(pattern)
            else:
                paths = dir_path.glob(pattern)
            
            for path in paths:
                # Skip hidden files if not included
                if not include_hidden and path.name.startswith('.'):
                    continue
                
                item_info = {
                    "name": path.name,
                    "path": str(path),
                    "type": "directory" if path.is_dir() else "file",
                    "size": path.stat().st_size if path.is_file() else None,
                    "modified": path.stat().st_mtime
                }
                items.append(item_info)
            
            # Sort by type (directories first) then by name
            items.sort(key=lambda x: (x["type"] != "directory", x["name"]))
            
            return ToolResult(
                success=True,
                data={
                    "directory": str(dir_path),
                    "pattern": pattern,
                    "recursive": recursive,
                    "total_items": len(items),
                    "items": items
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to list directory: {str(e)}"
            )


class CreateDirectoryTool(BaseTool):
    """Tool for creating directories"""
    
    @property
    def name(self) -> str:
        return "create_directory"
    
    @property
    def description(self) -> str:
        return "Create a new directory"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="directory_path",
                type="string",
                description="Path to the directory to create",
                required=True
            ),
            ToolParameter(
                name="parents",
                type="boolean",
                description="Create parent directories if they don't exist",
                required=False,
                default=True
            ),
            ToolParameter(
                name="exist_ok",
                type="boolean",
                description="Don't raise error if directory already exists",
                required=False,
                default=True
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute directory creation"""
        dir_path = Path(params["directory_path"])
        parents = params.get("parents", True)
        exist_ok = params.get("exist_ok", True)
        
        try:
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            
            return ToolResult(
                success=True,
                data={
                    "directory_path": str(dir_path),
                    "created": True
                }
            )
        except FileExistsError:
            return ToolResult(
                success=False,
                error=f"Directory already exists: {dir_path}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to create directory: {str(e)}"
            )


class MoveFileTool(BaseTool):
    """Tool for moving/renaming files"""
    
    @property
    def name(self) -> str:
        return "move_file"
    
    @property
    def description(self) -> str:
        return "Move or rename a file"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def requires_confirmation(self) -> bool:
        return True
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="source_path",
                type="string",
                description="Source file path",
                required=True
            ),
            ToolParameter(
                name="destination_path",
                type="string",
                description="Destination file path",
                required=True
            ),
            ToolParameter(
                name="overwrite",
                type="boolean",
                description="Overwrite destination if it exists",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute file move operation"""
        source = Path(params["source_path"])
        destination = Path(params["destination_path"])
        overwrite = params.get("overwrite", False)
        
        if not source.exists():
            return ToolResult(
                success=False,
                error=f"Source file not found: {source}"
            )
        
        if destination.exists() and not overwrite:
            return ToolResult(
                success=False,
                error=f"Destination already exists: {destination}. Set overwrite=True to overwrite."
            )
        
        try:
            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source), str(destination))
            
            return ToolResult(
                success=True,
                data={
                    "source_path": str(source),
                    "destination_path": str(destination),
                    "moved": True
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to move file: {str(e)}"
            )


class DeleteFileTool(BaseTool):
    """Tool for deleting files"""
    
    @property
    def name(self) -> str:
        return "delete_file"
    
    @property
    def description(self) -> str:
        return "Delete a file or directory"
    
    @property
    def category(self) -> str:
        return "file_operations"
    
    @property
    def requires_confirmation(self) -> bool:
        return True
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to delete",
                required=True
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="Delete directories recursively",
                required=False,
                default=False
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute delete operation"""
        path = Path(params["path"])
        recursive = params.get("recursive", False)
        
        if not path.exists():
            return ToolResult(
                success=False,
                error=f"Path not found: {path}"
            )
        
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
            
            return ToolResult(
                success=True,
                data={
                    "deleted_path": str(path),
                    "type": "directory" if path.is_dir() else "file"
                }
            )
        except OSError as e:
            return ToolResult(
                success=False,
                error=f"Failed to delete: {str(e)}"
            )