"""
OpenCode Repository Abstraction
Handles all file system operations for agents
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
import tempfile

logger = logging.getLogger('OpenCodeRepo')


class OpenCodeRepo:
    """
    Repository abstraction - agents never touch filesystem directly
    Supports both local editing and remote directory manipulation
    """
    
    def __init__(self, path: str, mode: str = "local"):
        """
        Initialize repository
        
        Args:
            path: Path to repository (can be any directory)
            mode: "local" (copy to temp) or "direct" (edit in place)
        """
        self.original_path = Path(path).resolve()
        self.mode = mode
        
        if not self.original_path.exists():
            raise ValueError(f"Repository path does not exist: {path}")
        
        if mode == "local":
            # Create temporary copy
            self.temp_dir = Path(tempfile.mkdtemp(prefix='opencode_'))
            self.root = self.temp_dir / 'repo'
            shutil.copytree(self.original_path, self.root)
            logger.info(f"Created local copy: {self.original_path} → {self.root}")
        else:
            # Edit directly in target directory
            self.temp_dir = None
            self.root = self.original_path
            logger.info(f"Direct mode: editing {self.root}")
    
    # ========================================================================
    # FILE OPERATIONS
    # ========================================================================
    
    def read(self, file_path: str) -> str:
        """Read file content"""
        full_path = self.root / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return full_path.read_text()
    
    def write(self, file_path: str, content: str) -> None:
        """Write file content"""
        full_path = self.root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        logger.info(f"Written: {file_path}")
    
    def delete(self, file_path: str) -> None:
        """Delete file"""
        full_path = self.root / file_path
        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted: {file_path}")
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists"""
        return (self.root / file_path).exists()
    
    # ========================================================================
    # DIRECTORY OPERATIONS
    # ========================================================================
    
    def list_files(self, pattern: str = "**/*.py", 
                   exclude_tests: bool = False) -> List[Path]:
        """
        List files matching pattern
        
        Args:
            pattern: Glob pattern
            exclude_tests: Exclude test files (default: False for flexibility)
        
        Returns:
            List of relative paths
        """
        files = []
        for file_path in self.root.glob(pattern):
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            
            # Skip tests if requested
            if exclude_tests and 'test' in str(file_path).lower():
                continue
            
            files.append(file_path.relative_to(self.root))
        
        logger.info(f"Found {len(files)} files matching {pattern}")
        return files
    
    def get_tree_structure(self, max_depth: int = 3) -> str:
        """Get repository tree structure"""
        lines = []
        
        def walk(path: Path, prefix: str = "", depth: int = 0):
            if depth >= max_depth:
                return
            
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            for i, item in enumerate(items):
                if item.name.startswith('.'):
                    continue
                
                is_last = i == len(items) - 1
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                
                if item.is_dir():
                    walk(item, prefix + ('    ' if is_last else '│   '), depth + 1)
        
        walk(self.root)
        return '\n'.join(lines)
    
    # ========================================================================
    # PATCH OPERATIONS
    # ========================================================================
    
    def _clean_patch_content(self, patch_content: str) -> Optional[str]:
        """
        Clean patch content by removing markdown fences and explanatory text.
        Extracts only the actual unified diff portion.
        
        Args:
            patch_content: Raw patch content (may include markdown/explanations)
        
        Returns:
            Cleaned patch content or None if no valid patch found
        """
        lines = patch_content.split('\n')
        
        # Find the start of the actual diff
        diff_start = None
        for i, line in enumerate(lines):
            # Look for lines that start a unified diff
            if line.startswith('--- ') or line.startswith('diff --git'):
                diff_start = i
                break
        
        if diff_start is None:
            logger.error("No diff header found in patch content")
            return None
        
        # Find the end of the diff (before any closing markdown fences)
        diff_end = len(lines)
        for i in range(diff_start + 1, len(lines)):
            line = lines[i].strip()
            # Stop at markdown code fence
            if line == '```' or line.startswith('```'):
                diff_end = i
                break
        
        # Extract and rejoin the diff portion
        diff_lines = lines[diff_start:diff_end]
        
        # Remove any remaining markdown artifacts
        cleaned_lines = []
        for line in diff_lines:
            # Skip lines that are just markdown fences
            if line.strip() in ['```', '```diff', '```patch']:
                continue
            cleaned_lines.append(line)
        
        if not cleaned_lines:
            logger.error("No valid diff content after cleaning")
            return None
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Validate it looks like a diff
        if not ('---' in cleaned and '+++' in cleaned):
            logger.error("Cleaned content doesn't look like a valid diff")
            return None
        
        return cleaned
    
    def apply_patch(self, patch_content: str) -> bool:
        """
        Apply unified diff patch using standard git/patch tools
        
        Args:
            patch_content: Unified diff format patch
        
        Returns:
            True if successful
        """
        try:
            import tempfile
            
            logger.info(f"Attempting to apply patch ({len(patch_content)} chars)")
            
            # Clean the patch content - remove markdown fences and preamble
            cleaned_patch = self._clean_patch_content(patch_content)
            
            if not cleaned_patch:
                logger.error("No valid patch content found after cleaning")
                return False
            
            logger.info(f"Cleaned patch ({len(cleaned_patch)} chars)")
            
            # Save patch to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                patch_file = Path(f.name)
                f.write(cleaned_patch)
            
            try:
                # Check if this is a git repo
                is_git_repo = (self.root / '.git').exists()
                
                if is_git_repo:
                    # Try git apply first (best option for git repos)
                    logger.info("Trying git apply...")
                    result = subprocess.run(
                        ['git', 'apply', '--verbose', str(patch_file)],
                        cwd=self.root,
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,  # Prevent interactive prompts
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✓ Patch applied successfully with git apply")
                        return True
                    else:
                        logger.warning(f"git apply failed: {result.stderr[:500]}")
                
                # Try patch command with multiple strip levels
                # For installed packages like pandas, -p2 is most common
                # -p2: strip 2 levels (a/pandas/path/file.py -> path/file.py)
                # -p1: strip 1 level (a/path/file.py -> path/file.py)
                # -p0: no stripping (a/path/file.py -> a/path/file.py)
                for strip_level in [2, 1, 0]:
                    logger.info(f"Trying patch command with -p{strip_level}...")
                    
                    # First do a dry-run to test if this will work
                    dry_run_result = subprocess.run(
                        ['patch', f'-p{strip_level}', '--dry-run', '-f', '--no-backup-if-mismatch', '-i', str(patch_file)],
                        cwd=self.root,
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,
                        timeout=30
                    )
                    
                    if dry_run_result.returncode != 0:
                        logger.warning(f"patch -p{strip_level} dry-run failed: {dry_run_result.stderr[:300]}")
                        continue
                    
                    # Dry-run succeeded, now apply for real
                    logger.info(f"Dry-run succeeded, applying patch with -p{strip_level}...")
                    result = subprocess.run(
                        ['patch', f'-p{strip_level}', '-f', '--no-backup-if-mismatch', '-i', str(patch_file)],
                        cwd=self.root,
                        capture_output=True,
                        text=True,
                        stdin=subprocess.DEVNULL,  # Prevent interactive prompts
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"✓ Patch applied successfully with patch -p{strip_level}")
                        # Clean up any .rej files that might exist
                        self._cleanup_reject_files()
                        return True
                    
                    logger.warning(f"patch -p{strip_level} failed: {result.stderr[:500]}")
                
                logger.error(f"All patch methods failed. Last error: {result.stderr[:500]}")
                # Clean up any .rej files
                self._cleanup_reject_files()
                return False
                
            except subprocess.TimeoutExpired:
                logger.error("⏱️  Patch application timed out after 30 seconds")
                return False
            except FileNotFoundError as e:
                logger.error(f"Command not found: {e}. Install git or patch tool.")
                return False
            finally:
                patch_file.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Patch application failed: {e}", exc_info=True)
            return False
    
    def _cleanup_reject_files(self):
        """Clean up any .rej or .orig files created by failed patch attempts"""
        try:
            for rej_file in self.root.rglob('*.rej'):
                rej_file.unlink()
                logger.debug(f"Cleaned up reject file: {rej_file}")
            for orig_file in self.root.rglob('*.orig'):
                orig_file.unlink()
                logger.debug(f"Cleaned up orig file: {orig_file}")
        except Exception as e:
            logger.debug(f"Error cleaning up reject files: {e}")
    
    def create_patch(self) -> Optional[str]:
        """
        Create patch of current changes
        
        Returns:
            Unified diff or None
        """
        try:
            result = subprocess.run(
                ['git', 'diff', '--no-color'],
                cwd=self.root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating patch: {e}")
            return None
    
    # ========================================================================
    # CODE ANALYSIS
    # ========================================================================
    
    def find_main_entry(self) -> Optional[Path]:
        """Find main entry point"""
        candidates = [
            'main.py',
            'app.py',
            'run.py',
            '__main__.py',
            'src/main.py',
        ]
        
        for candidate in candidates:
            path = self.root / candidate
            if path.exists():
                return path.relative_to(self.root)
        
        # Look for any .py file with if __name__ == '__main__'
        for py_file in self.root.glob('**/*.py'):
            try:
                content = py_file.read_text()
                if '__name__' in content and '__main__' in content:
                    return py_file.relative_to(self.root)
            except:
                continue
        
        return None
    
    def read_all_code(self, max_size: int = 100000) -> Dict[str, str]:
        """
        Read all code files
        
        Args:
            max_size: Maximum total size in characters
        
        Returns:
            Dict of {file_path: content}
        """
        code_files = {}
        total_size = 0
        
        for file_path in self.list_files():
            if total_size >= max_size:
                break
            
            try:
                content = self.read(str(file_path))
                if total_size + len(content) <= max_size:
                    code_files[str(file_path)] = content
                    total_size += len(content)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        logger.info(f"Read {len(code_files)} files ({total_size} chars)")
        return code_files
    
    # ========================================================================
    # SYNC OPERATIONS
    # ========================================================================
    
    def sync_to_original(self) -> None:
        """Sync changes back to original directory (for local mode)"""
        if self.mode != "local":
            logger.warning("sync_to_original only works in local mode")
            return
        
        if not self.temp_dir:
            logger.error("No temporary directory to sync from")
            return
        
        logger.info(f"Syncing {self.root} → {self.original_path}")
        
        # Remove original
        if self.original_path.exists():
            shutil.rmtree(self.original_path)
        
        # Copy back
        shutil.copytree(self.root, self.original_path)
        logger.info("✓ Synced to original directory")
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up: {self.temp_dir}")
    
    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()