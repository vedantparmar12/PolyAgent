from .base_tool import BaseTool
import os

class WriteFileTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    
    def execute(self, path: str, content: str) -> dict:
        try:
            # Get absolute path
            abs_path = os.path.abspath(path)
            
            # Create parent directories if needed
            parent_dir = os.path.dirname(abs_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Write file atomically using temporary file
            temp_path = abs_path + '.tmp'
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Atomic rename
                os.rename(temp_path, abs_path)
                
                return {
                    "path": abs_path,
                    "bytes_written": len(content.encode('utf-8')),
                    "success": True,
                    "message": f"Successfully wrote to {path}"
                }
            
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
        
        except PermissionError:
            return {"error": f"Permission denied writing to file: {path}"}
        except OSError as e:
            return {"error": f"OS error writing file: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}