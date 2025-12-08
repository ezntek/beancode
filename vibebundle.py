#!/usr/bin/env python3
"""
NOTE: THIS PROGRAM IS WRITTEN ALMOST ENTIRELY BY DEEPSEEK R1 AS A BUNDLER FOR BEANCODE WEB.
This is a crucial step to run beancode in the web, to combine everything into a single file to reduce
file size.

therefore, this is put under the public domain.

---

Python Module Bundler - Bundles multiple Python files in a module into a single file.
Preserves correct import order with dependencies before dependents.
"""

import ast
import sys
from pathlib import Path
import re
import importlib.util

class PythonBundler:
    def __init__(self, root_module_path: str, output_file: str | None = None, comments: bool = True):
        """
        Initialize the bundler.
        
        Args:
            root_module_path: Path to the module directory or entry point file
            output_file: Output file path (optional)
        """
        self.comments = comments
        self.root_path = Path(root_module_path).resolve()
        self.output_file = Path(output_file) if output_file else None
        
        # Track all files in dependency order
        self.processed_files: set[Path] = set()
        self.files_in_order: list[Path] = []
        
        # Import tracking
        self.external_imports: set[str] = set()
        self.local_import_map: dict[Path, set[Path]] = {}  # file -> files it imports
        self.depended_by_map: dict[Path, set[Path]] = {}  # file -> files that import it
        
        # File content cache
        self.file_content: dict[Path, str] = {}
        
        # Module root detection
        self.module_root: Path | None = None
        
    def find_module_root(self, start_path: Path) -> Path:
        """
        Find the root of the Python module (directory with __init__.py).
        """
        current = start_path if start_path.is_dir() else start_path.parent
        
        # Walk up until we find a directory without __init__.py in parent
        while current:
            init_py = current / "__init__.py"
            parent = current.parent
            parent_init_py = parent / "__init__.py" if parent else None
            
            if init_py.exists():
                if not parent_init_py or not parent_init_py.exists():
                    return current
            current = parent
        
        # If no clear module root, use the directory containing the entry point
        return self.root_path if self.root_path.is_dir() else self.root_path.parent
    
    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is from the standard library."""
        # Extract base module name
        base_module = module_name.split('.')[0]
        
        # Check built-in modules
        if base_module in sys.builtin_module_names:
            return True
        
        # Common stdlib modules
        common_stdlib = {
            'os', 'sys', 're', 'json', 'collections', 'itertools', 'functools',
            'datetime', 'time', 'math', 'random', 'string', 'pathlib', 'typing',
            'subprocess', 'shutil', 'argparse', 'hashlib', 'base64', 'csv',
            'io', 'pprint', 'inspect', 'textwrap', 'html', 'xml', 'uuid',
            'socket', 'ssl', 'http', 'urllib', 'email', 'zipfile', 'tarfile',
            'pickle', 'shelve', 'sqlite3', 'logging', 'unittest', 'doctest',
            'decimal', 'fractions', 'statistics', 'copy', 'pprint', 'enum',
            'dataclasses', 'contextlib', 'abc', 'weakref', 'types', 'warnings',
            'traceback', 'linecache', 'tokenize', 'keyword', 'dis', 'ast'
        }
        
        if base_module in common_stdlib:
            return True
        
        # Try to import and check location
        try:
            spec = importlib.util.find_spec(base_module)
            if spec and spec.origin:
                # Check if it's in stdlib paths
                stdlib_paths = [sys.prefix, sys.exec_prefix]
                for path in stdlib_paths:
                    if spec.origin.startswith(str(path)):
                        return True
        except (ImportError, ValueError):
            pass
        
        return False
    
    def is_local_import(self, import_path: str, current_file: Path) -> bool:
        """
        Check if an import is local to the module.
        """
        if not import_path:
            return False
            
        # Handle relative imports - always local
        if import_path.startswith('.'):
            return True
            
        # Check if it's likely a stdlib module
        if self.is_stdlib_module(import_path):
            return False
            
        # Check if we can find it locally
        module_root = self.module_root or self.find_module_root(current_file)
        parts = import_path.split('.')
        
        # Check for .py file
        potential_file = module_root / Path(*parts).with_suffix('.py')
        if potential_file.exists():
            return True
            
        # Check for package
        potential_dir = module_root / Path(*parts)
        init_file = potential_dir / "__init__.py"
        if init_file.exists():
            return True
            
        # Check for parent directories
        for i in range(len(parts) - 1):
            parent_path = module_root / Path(*parts[:i+1])
            child_file = parent_path / f"{parts[i+1]}.py"
            if child_file.exists():
                return True
                
        return False
    
    def resolve_import_to_path(self, import_path: str, current_file: Path) -> Path | None:
        """
        Resolve an import path to a file path.
        """
        if not import_path:
            return None
            
        module_root = self.module_root or self.find_module_root(current_file)
        
        # Handle relative imports
        if import_path.startswith('.'):
            # Count dots
            level = 0
            while import_path.startswith('.'):
                import_path = import_path[1:]
                level += 1
            
            # Start from current file's directory
            target_dir = current_file.parent
            for _ in range(level - 1):
                target_dir = target_dir.parent
            
            # If import_path is empty, we're importing current package
            if not import_path:
                # Check for __init__.py
                init_file = target_dir / "__init__.py"
                if init_file.exists():
                    return init_file
                return None
            
            parts = import_path.split('.')
            
            # Check for .py file
            py_file = target_dir / Path(*parts).with_suffix('.py')
            if py_file.exists():
                return py_file
            
            # Check for package
            pkg_dir = target_dir / Path(*parts)
            init_file = pkg_dir / "__init__.py"
            if init_file.exists():
                return init_file
            
            return None
        
        # Absolute import
        parts = import_path.split('.')
        
        # Check for .py file
        py_file = module_root / Path(*parts).with_suffix('.py')
        if py_file.exists():
            return py_file
        
        # Check for package
        pkg_dir = module_root / Path(*parts)
        init_file = pkg_dir / "__init__.py"
        if init_file.exists():
            return init_file
        
        # Check intermediate paths
        for i in range(len(parts) - 1):
            parent_dir = module_root / Path(*parts[:i+1])
            child_file = parent_dir / f"{parts[i+1]}.py"
            if child_file.exists():
                return child_file
        
        return None
    
    def extract_imports_from_ast(self, code: str) -> tuple[list[str], list[str]]:
        """
        Extract imports from code using AST.
        Returns: (local_imports, external_imports)
        """
        local_imports = []
        external_imports = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        # Store full import statement
                        if alias.asname:
                            external_imports.append(f"import {module_name} as {alias.asname}")
                        else:
                            external_imports.append(f"import {module_name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module
                        # Handle relative imports
                        if node.level > 0:
                            dots = '.' * node.level
                            full_module = f"{dots}{module_name}" if module_name else dots
                            local_imports.append(full_module)
                        else:
                            # Check if it's local or external
                            # We'll check later when we have the file context
                            imports_list = []
                            for alias in node.names:
                                if alias.asname:
                                    imports_list.append(f"{alias.name} as {alias.asname}")
                                else:
                                    imports_list.append(alias.name)
                            
                            import_stmt = f"from {module_name} import {', '.join(imports_list)}"
                            # Store for later classification
                            local_imports.append(import_stmt)
        
        except SyntaxError:
            # Fallback to regex for malformed code
            pass
            
        return local_imports, external_imports
    
    def extract_imports_from_content(self, code: str, current_file: Path) -> tuple[set[str], set[str]]:
        """
        Extract imports from code content.
        Returns: (local_import_paths, external_import_statements)
        """
        local_imports = set()
        external_imports = set()
        
        lines = code.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check for import statements
            if line.startswith('import ') or line.startswith('from '):
                # Collect multi-line import
                import_lines = [line]
                j = i + 1
                
                # Check for continuation
                while j < len(lines):
                    next_line = lines[j].rstrip()
                    if not next_line:
                        break
                    
                    # Check if line continues the import
                    if (next_line.endswith('\\') or 
                        next_line.endswith('(') or
                        (import_lines[-1].endswith(',') and not next_line.startswith(' ') and not next_line.startswith('\t'))):
                        import_lines.append(next_line)
                        j += 1
                    else:
                        # Check if this is part of the import
                        if (next_line.startswith(' ') or next_line.startswith('\t')):
                            import_lines.append(next_line)
                            j += 1
                        else:
                            break
                
                full_import = '\n'.join(import_lines)
                
                # Parse the import
                if full_import.startswith('import '):
                    # Simple import
                    match = re.match(r'^import\s+(\S+)(?:\s+as\s+\S+)?', full_import)
                    if match:
                        module_name = match.group(1)
                        if self.is_local_import(module_name, current_file):
                            local_imports.add(module_name)
                        else:
                            external_imports.add(full_import)
                
                elif full_import.startswith('from '):
                    # From import
                    match = re.match(r'^from\s+(\S+)\s+import', full_import)
                    if match:
                        module_name = match.group(1)
                        if module_name.startswith('.'):
                            # Relative import - always local
                            local_imports.add(module_name)
                        elif self.is_local_import(module_name, current_file):
                            local_imports.add(module_name)
                        else:
                            external_imports.add(full_import)
                
                i = j
            else:
                i += 1
        
        return local_imports, external_imports
    
    def analyze_file_dependencies(self, file_path: Path):
        """
        Analyze a file's dependencies and update dependency graphs.
        """
        if file_path in self.processed_files:
            return
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        self.file_content[file_path] = content
        
        # Extract imports
        local_imports, external_imports = self.extract_imports_from_content(content, file_path)
        
        # Store external imports
        self.external_imports.update(external_imports)
        
        # Initialize dependency sets
        self.local_import_map[file_path] = set()
        if file_path not in self.depended_by_map:
            self.depended_by_map[file_path] = set()
        
        # Process local imports
        for import_path in local_imports:
            imported_file = self.resolve_import_to_path(import_path, file_path)
            if imported_file and imported_file.exists():
                self.local_import_map[file_path].add(imported_file)
                
                # Update depended_by map
                if imported_file not in self.depended_by_map:
                    self.depended_by_map[imported_file] = set()
                self.depended_by_map[imported_file].add(file_path)
    
    def find_entry_point(self) -> Path:
        """
        Find the entry point file (__main__.py or __init__.py).
        """
        if self.root_path.is_file():
            return self.root_path
        
        # Look for __main__.py
        main_file = self.root_path / "__main__.py"
        if main_file.exists():
            return main_file
        
        # Look for __init__.py
        init_file = self.root_path / "__init__.py"
        if init_file.exists():
            return init_file
        
        # Find any .py file
        py_files = list(self.root_path.glob("*.py"))
        if py_files:
            return py_files[0]
        
        raise FileNotFoundError(f"No Python files found in {self.root_path}")
    
    def collect_all_files(self, start_file: Path) -> list[Path]:
        """
        Collect all files in dependency order (dependencies before dependents).
        Uses topological sort based on import dependencies.
        """
        # First, find all files reachable from start_file
        files_to_process = [start_file]
        all_files = set()
        
        while files_to_process:
            current_file = files_to_process.pop()
            
            if current_file in all_files:
                continue
            
            all_files.add(current_file)
            
            # Analyze this file's dependencies
            self.analyze_file_dependencies(current_file)
            
            # Add its dependencies to processing queue
            for dep in self.local_import_map.get(current_file, set()):
                if dep not in all_files:
                    files_to_process.append(dep)
        
        # Now order files topologically
        ordered_files = []
        visited = set()
        temp_visited = set()
        
        def visit(file_path: Path):
            if file_path in temp_visited:
                # Circular dependency detected
                return
            if file_path in visited:
                return
            
            temp_visited.add(file_path)
            
            # Visit dependencies first
            for dep in self.local_import_map.get(file_path, set()):
                if dep in all_files:  # Only visit files in our module
                    visit(dep)
            
            temp_visited.remove(file_path)
            visited.add(file_path)
            ordered_files.append(file_path)
        
        # Start with files that have no local imports
        for file_path in all_files:
            if not self.local_import_map.get(file_path, set()):
                visit(file_path)
        
        # Then visit remaining files
        for file_path in all_files:
            if file_path not in visited:
                visit(file_path)
        
        # Ensure __init__.py files come first if they exist
        init_files = [f for f in ordered_files if f.name == "__init__.py"]
        other_files = [f for f in ordered_files if f.name != "__init__.py"]
        
        # Sort init files by depth (shallow first)
        init_files.sort(key=lambda f: len(f.parts))
        
        return init_files + other_files
    
    def remove_imports_from_content(self, content: str) -> str:
        """
        Remove import statements from content.
        """
        lines = content.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if not line or line.startswith('#'):
                result.append(lines[i])
                i += 1
                continue
            
            # Check for import
            if line.startswith('import ') or line.startswith('from '):
                # Skip this import
                i += 1
                # Skip continuation lines
                while i < len(lines):
                    next_line = lines[i].rstrip()
                    if (next_line.endswith('\\') or 
                        next_line.endswith('(') or
                        (i > 0 and lines[i-1].rstrip().endswith(','))):
                        i += 1
                    else:
                        break
            else:
                result.append(lines[i])
                i += 1
        
        return '\n'.join(result)
    
    def process_file_for_output(self, file_path: Path) -> str:
        """
        Process a file for final output (remove imports, clean up).
        """
        content = self.file_content[file_path]
        
        # Remove imports
        content = self.remove_imports_from_content(content)
        
        # Add file header
        if file_path.parent == self.module_root:
            rel_path = file_path.name
        else:
            rel_path = file_path.relative_to(self.module_root.parent if self.module_root.parent else self.module_root) # type: ignore
       
        header = ""
        if self.comments:
            header = f"\n{'#' * 60}\n# File: {rel_path}\n{'#' * 60}\n"
        
        return header + content + "\n"
    
    def bundle(self) -> str:
        """
        Bundle the module into a single file.
        """
        # Find module root
        self.module_root = self.find_module_root(self.root_path)
        
        # Find entry point
        entry_point = self.find_entry_point()
        
        # Collect all files in dependency order
        all_files = self.collect_all_files(entry_point)
        
        # Ensure __init__.py at module root is included if it exists
        root_init = self.module_root / "__init__.py"
        if root_init.exists() and root_init not in all_files:
            # Insert at beginning
            all_files.insert(0, root_init)
            self.analyze_file_dependencies(root_init)
        
        # Build output
        output_parts = []
        
        # Header
        if self.comments:
            output_parts.append("#" * 70)
            output_parts.append("# Bundled Python Module")
            output_parts.append(f"# Generated from: {entry_point}")
            output_parts.append(f"# Module root: {self.module_root}")
            output_parts.append("#" * 70)
            output_parts.append("")
        
        # External imports
        if self.external_imports:
            if self.comments:
                output_parts.append("#" * 60)
                output_parts.append("# External Imports")
                output_parts.append("#" * 60)
            
            # Sort imports
            sorted_imports = sorted(self.external_imports, key=lambda x: (
                0 if 'from __future__' in x else
                1 if x.startswith('import ') else 2,
                x.lower()
            ))
            
            for imp in sorted_imports:
                output_parts.append(imp)
            
            output_parts.append("")
        
        # Module content in dependency order
        if self.comments:
            output_parts.append("#" * 60)
            output_parts.append("# Module Content (in dependency order)")
            output_parts.append("#" * 60)
        
        for file_path in all_files:
            output_parts.append(self.process_file_for_output(file_path))
        
        return '\n'.join(output_parts)
    
    def save(self, code: str | None = None) -> Path:
        """
        Save the bundled code to a file.
        """
        if code is None:
            code = self.bundle()
        
        if self.output_file:
            output_path = self.output_file
        else:
            base_name = self.module_root.name if self.module_root else "bundled"
            output_path = (self.module_root.parent if self.module_root else Path.cwd()) / f"{base_name}_bundled.py"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"Saved to: {output_path}")
        print(f"Output size: {len(code.splitlines())} lines")
        
        return output_path


def main():
    """Command-line interface."""

    print("""\033[1mNOTE: THIS PROGRAM IS WRITTEN ALMOST ENTIRELY BY DEEPSEEK R1 AS A BUNDLER FOR BEANCODE WEB.
This is a crucial step to run beancode in the web, to combine everything into a single file to reduce
file size.

Therefore, this is put under the public domain.
\033[0m""")

    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bundle Python module into single file with correct dependency ordering"
    )
    parser.add_argument(
        "module",
        help="Path to module directory or entry point Python file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <module_name>_bundled.py)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print bundled code to stdout instead of saving"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    parser.add_argument(
        "--comments",
        action="store_true",
        help="Write comments in the file"
    )
    
    args = parser.parse_args()

    try:
        bundler = PythonBundler(args.module, args.output, args.comments)
        
        bundled_code = bundler.bundle()
        
        if args.dry_run:
            print(bundled_code)
        else:
            bundler.save(bundled_code)
            
            if args.verbose:
                print("\nFiles processed in order:")
                for i, file_path in enumerate(bundler.files_in_order, 1):
                    print(f"  {i:2d}. {file_path.relative_to(file_path.parent.parent) if file_path.parent.parent else file_path.name}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
