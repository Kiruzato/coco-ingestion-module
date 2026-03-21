#!/usr/bin/env python3
"""
CoCo Ingestion Module — Desktop GUI
====================================
Provides a graphical interface for:
  - Configuring and verifying the OpenAI API key
  - Selecting a folder of documents to ingest
  - Running the ingestion process
  - Viewing progress and results
"""

import sys
import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

# Setup path so module imports work
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

CONFIG_FILE = SCRIPT_DIR / "ingestion_config.json"


# ============================================================================
# Configuration Persistence
# ============================================================================

def load_config() -> dict:
    """Load saved configuration from disk."""
    defaults = {"openai_api_key": "", "last_folder": ""}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_config(config: dict):
    """Persist configuration to disk."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save config: {e}")


# ============================================================================
# API Key Verification
# ============================================================================

def verify_openai_key(api_key: str) -> tuple[bool, str]:
    """
    Verify an OpenAI API key by making a real API call.

    Returns:
        (is_valid, message)
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        client.models.list()
        return True, "API key is valid."
    except Exception as e:
        err = str(e)
        if "auth" in err.lower() or "key" in err.lower() or "401" in err:
            return False, "Invalid API key."
        return False, f"Verification failed: {err}"


# ============================================================================
# GUI Application
# ============================================================================

class IngestionApp:
    """Main tkinter application for the CoCo Ingestion Module."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CoCo Ingestion Module")
        self.root.geometry("700x580")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f5f5")

        self.config = load_config()
        self.selected_folder = self.config.get("last_folder", "")
        self.is_running = False

        self._build_ui()
        self._load_saved_state()

    # ----------------------------------------------------------------
    # UI Construction
    # ----------------------------------------------------------------

    def _build_ui(self):
        """Build all UI components."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"),
                        background="#f5f5f5", foreground="#1e3c72")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))
        style.configure("Run.TButton", font=("Segoe UI", 12, "bold"), padding=10)

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main_frame, text="CoCo Ingestion Module",
                  style="Title.TLabel").pack(pady=(0, 15))

        # --- API Key Section ---
        api_frame = ttk.LabelFrame(main_frame, text="  OpenAI API Key  ",
                                   padding=15, style="Section.TLabelframe")
        api_frame.pack(fill=tk.X, pady=(0, 10))

        key_row = ttk.Frame(api_frame)
        key_row.pack(fill=tk.X)

        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(key_row, textvariable=self.api_key_var,
                                       show="*", width=50)
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.toggle_key_btn = ttk.Button(key_row, text="Show", width=6,
                                         command=self._toggle_key_visibility)
        self.toggle_key_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.verify_btn = ttk.Button(key_row, text="Verify", width=8,
                                     command=self._verify_key)
        self.verify_btn.pack(side=tk.LEFT)

        self.key_status_var = tk.StringVar(value="")
        self.key_status_label = ttk.Label(api_frame, textvariable=self.key_status_var,
                                          style="Status.TLabel")
        self.key_status_label.pack(anchor=tk.W, pady=(8, 0))

        # --- Folder Selection Section ---
        folder_frame = ttk.LabelFrame(main_frame, text="  Document Folder  ",
                                      padding=15, style="Section.TLabelframe")
        folder_frame.pack(fill=tk.X, pady=(0, 10))

        folder_row = ttk.Frame(folder_frame)
        folder_row.pack(fill=tk.X)

        self.folder_var = tk.StringVar(value="No folder selected")
        self.folder_label = ttk.Label(folder_row, textvariable=self.folder_var,
                                      width=55, anchor=tk.W, relief="sunken",
                                      padding=(5, 4))
        self.folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.browse_btn = ttk.Button(folder_row, text="Browse...", width=10,
                                     command=self._browse_folder)
        self.browse_btn.pack(side=tk.LEFT)

        self.folder_info_var = tk.StringVar(value="")
        ttk.Label(folder_frame, textvariable=self.folder_info_var,
                  style="Status.TLabel").pack(anchor=tk.W, pady=(8, 0))

        # --- Ingestion Button ---
        self.run_btn = ttk.Button(main_frame, text="Start Ingestion",
                                  style="Run.TButton", command=self._start_ingestion)
        self.run_btn.pack(fill=tk.X, pady=(5, 10))

        # --- Progress Section ---
        progress_frame = ttk.LabelFrame(main_frame, text="  Progress  ",
                                        padding=15, style="Section.TLabelframe")
        progress_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        self.log_text = tk.Text(progress_frame, height=10, font=("Consolas", 9),
                                bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
                                relief="flat", padx=8, pady=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var,
                  style="Status.TLabel").pack(anchor=tk.W, pady=(8, 0))

    # ----------------------------------------------------------------
    # State Management
    # ----------------------------------------------------------------

    def _load_saved_state(self):
        """Restore saved state from config."""
        key = self.config.get("openai_api_key", "")
        if key:
            self.api_key_var.set(key)
            self.key_status_var.set("Key loaded from saved config.")
            self.key_status_label.configure(foreground="#2e7d32")

        folder = self.config.get("last_folder", "")
        if folder and Path(folder).is_dir():
            self.selected_folder = folder
            self.folder_var.set(folder)
            self._update_folder_info(Path(folder))

    def _save_current_state(self):
        """Save current state to config."""
        self.config["openai_api_key"] = self.api_key_var.get().strip()
        self.config["last_folder"] = self.selected_folder
        save_config(self.config)

    # ----------------------------------------------------------------
    # API Key Handling
    # ----------------------------------------------------------------

    def _toggle_key_visibility(self):
        if self.api_key_entry.cget("show") == "*":
            self.api_key_entry.configure(show="")
            self.toggle_key_btn.configure(text="Hide")
        else:
            self.api_key_entry.configure(show="*")
            self.toggle_key_btn.configure(text="Show")

    def _verify_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            self.key_status_var.set("Please enter an API key.")
            self.key_status_label.configure(foreground="#c62828")
            return

        self.verify_btn.configure(state=tk.DISABLED)
        self.key_status_var.set("Verifying...")
        self.key_status_label.configure(foreground="#1565c0")

        def do_verify():
            valid, msg = verify_openai_key(key)
            self.root.after(0, lambda: self._on_verify_done(valid, msg))

        threading.Thread(target=do_verify, daemon=True).start()

    def _on_verify_done(self, valid: bool, msg: str):
        self.verify_btn.configure(state=tk.NORMAL)
        self.key_status_var.set(msg)
        if valid:
            self.key_status_label.configure(foreground="#2e7d32")
            self._save_current_state()
        else:
            self.key_status_label.configure(foreground="#c62828")

    # ----------------------------------------------------------------
    # Folder Selection
    # ----------------------------------------------------------------

    def _browse_folder(self):
        initial = self.selected_folder if self.selected_folder else str(Path.home())
        folder = filedialog.askdirectory(title="Select Document Folder",
                                         initialdir=initial)
        if folder:
            self.selected_folder = folder
            self.folder_var.set(folder)
            self._update_folder_info(Path(folder))
            self._save_current_state()

    def _update_folder_info(self, folder: Path):
        """Count and display supported documents in the folder."""
        exts = {'.txt', '.pdf', '.docx'}
        files = [f for f in folder.iterdir()
                 if f.is_file() and f.suffix.lower() in exts]
        self.folder_info_var.set(f"Found {len(files)} supported document(s) (.txt, .pdf, .docx)")

    # ----------------------------------------------------------------
    # Ingestion
    # ----------------------------------------------------------------

    def _log(self, text: str):
        """Append text to the log panel (thread-safe via root.after)."""
        def _do():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _do)

    def _start_ingestion(self):
        # Validate inputs
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showwarning("Missing API Key",
                                   "Please enter your OpenAI API key.")
            return

        if not self.selected_folder or not Path(self.selected_folder).is_dir():
            messagebox.showwarning("No Folder Selected",
                                   "Please select a folder containing documents.")
            return

        if self.is_running:
            return

        self.is_running = True
        self.run_btn.configure(state=tk.DISABLED)
        self.browse_btn.configure(state=tk.DISABLED)
        self.verify_btn.configure(state=tk.DISABLED)
        self.progress_bar.start(15)
        self.status_var.set("Ingestion in progress...")

        # Clear log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        # Save key to environment and config before running
        os.environ["OPENAI_API_KEY"] = key
        self._save_current_state()

        threading.Thread(target=self._run_ingestion, daemon=True).start()

    def _run_ingestion(self):
        """Run the ingestion process in a background thread.

        Matches the behavior of ingest.py (INGESTION_MODULE-original):
        - Creates a timestamped package folder inside the documents folder
        - Builds vector store and registry inside the package folder
        - Creates a .zip alongside the package folder (inside documents folder)
        - Source files are only read, never modified or deleted
        """
        try:
            from dotenv import load_dotenv
            load_dotenv(SCRIPT_DIR / ".env")
            # Ensure the GUI-entered key takes precedence
            os.environ["OPENAI_API_KEY"] = self.api_key_var.get().strip()

            from modules.document_manager import DocumentManager
            from modules.config import (
                DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
                REQUIRED_VECTOR_STORE_FILES
            )

            documents_path = Path(self.selected_folder)
            self._log(f"Documents folder: {documents_path}")

            # Count documents (exclude rag_package_* folders from previous runs)
            doc_files = [
                f for f in documents_path.glob("**/*")
                if f.is_file()
                and f.suffix.lower() in {'.txt', '.pdf', '.docx'}
                and 'rag_package_' not in str(f)
            ]
            self._log(f"Found {len(doc_files)} document(s)")

            if not doc_files:
                self._log("ERROR: No supported documents found.")
                self.root.after(0, self._on_ingestion_done, False, "No documents found.")
                return

            # Create timestamped package folder inside the documents folder
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            package_name = f"rag_package_{timestamp}"
            package_dir = documents_path / package_name
            package_dir.mkdir(parents=True, exist_ok=True)

            vector_store_dir = package_dir / "vector_store"
            registry_path = package_dir / "document_registry.json"

            self._log(f"Output folder: {package_dir}")
            self._log("")
            self._log("--- Ingesting Documents ---")

            manager = DocumentManager(
                registry_path=registry_path,
                vector_store_path=vector_store_dir
            )

            results = manager.ingest_directory(
                directory_path=documents_path,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                skip_duplicates=True
            )

            manager.save_vector_store()

            success_count = len(results.get("success", []))
            skipped_count = len(results.get("skipped", []))
            failed_count = len(results.get("failed", []))

            self._log("")
            self._log("--- Ingestion Summary ---")
            self._log(f"  Successful: {success_count}")
            self._log(f"  Skipped:    {skipped_count}")
            self._log(f"  Failed:     {failed_count}")

            # Validate output
            self._log("")
            self._log("--- Validating Output ---")
            all_valid = True
            for req in REQUIRED_VECTOR_STORE_FILES:
                fp = vector_store_dir / req
                if fp.exists():
                    size = fp.stat().st_size
                    size_str = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.1f} MB"
                    self._log(f"  [OK] {req} ({size_str})")
                else:
                    self._log(f"  [MISSING] {req}")
                    all_valid = False

            if registry_path.exists():
                size = registry_path.stat().st_size
                size_str = f"{size/1024:.1f} KB"
                self._log(f"  [OK] document_registry.json ({size_str})")
            else:
                self._log(f"  [MISSING] document_registry.json")
                all_valid = False

            if not all_valid:
                self._log("")
                self._log("ERROR: Package validation failed!")
                self.root.after(0, self._on_ingestion_done, False, "Validation failed.")
                return

            # Create ZIP in the documents folder (alongside the package folder)
            self._log("")
            self._log("--- Creating ZIP Package ---")
            import zipfile
            zip_path = documents_path / f"{package_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for fp in vector_store_dir.rglob("*"):
                    if fp.is_file():
                        arcname = f"vector_store/{fp.relative_to(vector_store_dir)}"
                        zipf.write(fp, arcname)
                if registry_path.exists():
                    zipf.write(registry_path, "document_registry.json")

            zip_size = zip_path.stat().st_size
            zip_str = f"{zip_size/1024:.1f} KB" if zip_size < 1048576 else f"{zip_size/1048576:.1f} MB"
            self._log(f"  Created: {zip_path.name} ({zip_str})")

            self._log("")
            self._log("=" * 50)
            self._log("  Package Ready!")
            self._log("=" * 50)
            self._log(f"  Folder: {package_dir}")
            self._log(f"  ZIP:    {zip_path}")

            self.root.after(0, self._on_ingestion_done, True,
                           f"Package created: {zip_path}")

        except Exception as e:
            self._log(f"\nERROR: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.root.after(0, self._on_ingestion_done, False, str(e))

    def _on_ingestion_done(self, success: bool, message: str):
        """Called on the main thread when ingestion completes."""
        self.is_running = False
        self.progress_bar.stop()
        self.run_btn.configure(state=tk.NORMAL)
        self.browse_btn.configure(state=tk.NORMAL)
        self.verify_btn.configure(state=tk.NORMAL)

        if success:
            self.status_var.set("Ingestion completed successfully.")
            messagebox.showinfo("Success", message)
        else:
            self.status_var.set(f"Ingestion failed: {message}")
            messagebox.showerror("Error", f"Ingestion failed:\n{message}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    root = tk.Tk()
    app = IngestionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
