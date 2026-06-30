#!/usr/bin/env python3
"""Simple desktop launcher for tes_encoder.exe."""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


DEFAULT_GRID_QUALITY = "128"
DEFAULT_ALPHA = "0.999"
DEFAULT_EPSILON = "1e-6"


class TesEncoderUi:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TES Encoder UI")
        self.root.geometry("980x720")
        self.root.minsize(840, 620)

        self.repo_root = Path(__file__).resolve().parents[1]
        default_exe = self.repo_root / "build" / "Release" / "tes_encoder.exe"

        self.encoder_path_var = tk.StringVar(value=str(default_exe))
        self.input_path_var = tk.StringVar()
        self.outer_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.grid_quality_var = tk.StringVar(value=DEFAULT_GRID_QUALITY)
        self.alpha_var = tk.StringVar(value=DEFAULT_ALPHA)
        self.epsilon_var = tk.StringVar(value=DEFAULT_EPSILON)
        self.deformable_var = tk.BooleanVar(value=False)
        self.command_preview_var = tk.StringVar(value="")

        self.last_msh_dir = self._default_existing_dir(self.repo_root / "example" / "MSH")
        self.last_tes_dir = self._default_existing_dir(self.repo_root / "example" / "TES")

        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._output_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._update_command_preview()
        self.root.after(100, self._drain_output_queue)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        row = self._add_path_row(
            parent=frame,
            row=row,
            label="Encoder executable",
            variable=self.encoder_path_var,
            browse_callback=self._browse_encoder,
            browse_text="Browse EXE",
        )
        row = self._add_path_row(
            parent=frame,
            row=row,
            label="Input mesh (.msh)",
            variable=self.input_path_var,
            browse_callback=self._browse_input,
            browse_text="Browse Input",
        )
        row = self._add_path_row(
            parent=frame,
            row=row,
            label="Outer mesh (.msh, optional)",
            variable=self.outer_path_var,
            browse_callback=self._browse_outer,
            browse_text="Browse Outer",
            clearable=True,
        )
        row = self._add_path_row(
            parent=frame,
            row=row,
            label="Output TES (.json)",
            variable=self.output_path_var,
            browse_callback=self._browse_output,
            browse_text="Browse Output",
        )

        options = ttk.LabelFrame(frame, text="Encoding options", padding=10)
        options.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Grid quality").grid(row=0, column=0, sticky="w")
        grid_quality_entry = ttk.Entry(options, textvariable=self.grid_quality_var, width=16)
        grid_quality_entry.grid(row=0, column=1, sticky="w", padx=(8, 24))

        ttk.Label(options, text="Alpha").grid(row=0, column=2, sticky="w")
        alpha_entry = ttk.Entry(options, textvariable=self.alpha_var, width=16)
        alpha_entry.grid(row=0, column=3, sticky="w", padx=(8, 0))

        ttk.Label(options, text="Epsilon").grid(row=1, column=0, sticky="w", pady=(8, 0))
        epsilon_entry = ttk.Entry(options, textvariable=self.epsilon_var, width=16)
        epsilon_entry.grid(row=1, column=1, sticky="w", padx=(8, 24), pady=(8, 0))

        deformable_check = ttk.Checkbutton(
            options,
            text="Deformable encoding",
            variable=self.deformable_var,
            command=self._update_command_preview,
        )
        deformable_check.grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))

        row += 1
        ttk.Label(frame, text="Command preview").grid(row=row, column=0, sticky="w", pady=(10, 4))
        row += 1

        preview_entry = ttk.Entry(frame, textvariable=self.command_preview_var, state="readonly")
        preview_entry.grid(row=row, column=0, columnspan=4, sticky="ew")

        row += 1
        button_bar = ttk.Frame(frame)
        button_bar.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        button_bar.columnconfigure(0, weight=0)
        button_bar.columnconfigure(1, weight=0)
        button_bar.columnconfigure(2, weight=1)

        self.run_button = ttk.Button(button_bar, text="Run encode", command=self._start_encode)
        self.run_button.grid(row=0, column=0, sticky="w")

        self.stop_button = ttk.Button(button_bar, text="Stop", command=self._stop_encode, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        clear_button = ttk.Button(button_bar, text="Clear log", command=self._clear_log)
        clear_button.grid(row=0, column=2, sticky="e")

        row += 1
        ttk.Label(frame, text="Output log").grid(row=row, column=0, sticky="w", pady=(10, 4))

        row += 1
        self.log_box = ScrolledText(frame, wrap=tk.WORD, height=22)
        self.log_box.grid(row=row, column=0, columnspan=4, sticky="nsew")

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=0)
        frame.rowconfigure(row, weight=1)

        tracked_vars = (
            self.encoder_path_var,
            self.input_path_var,
            self.outer_path_var,
            self.output_path_var,
            self.grid_quality_var,
            self.alpha_var,
            self.epsilon_var,
        )
        for var in tracked_vars:
            var.trace_add("write", lambda *_: self._update_command_preview())

        self.input_path_var.trace_add("write", lambda *_: self._update_last_msh_dir(self.input_path_var.get()))
        self.outer_path_var.trace_add("write", lambda *_: self._update_last_msh_dir(self.outer_path_var.get()))
        self.output_path_var.trace_add("write", lambda *_: self._update_last_tes_dir(self.output_path_var.get()))

        grid_quality_entry.focus_set()

    def _default_existing_dir(self, preferred: Path) -> Path:
        if preferred.exists() and preferred.is_dir():
            return preferred
        return self.repo_root

    def _extract_existing_dir(self, path_text: str) -> Path | None:
        text = path_text.strip()
        if not text:
            return None

        path = Path(text)
        if path.exists():
            if path.is_dir():
                return path
            return path.parent

        parent = path.parent
        if parent.exists() and parent.is_dir():
            return parent

        return None

    def _update_last_msh_dir(self, path_text: str) -> None:
        existing_dir = self._extract_existing_dir(path_text)
        if existing_dir is not None:
            self.last_msh_dir = existing_dir

    def _update_last_tes_dir(self, path_text: str) -> None:
        existing_dir = self._extract_existing_dir(path_text)
        if existing_dir is not None:
            self.last_tes_dir = existing_dir

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_callback,
        browse_text: str,
        clearable: bool = False,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 8))

        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(0, 8))

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=3, sticky="e", pady=(0, 8))

        browse_button = ttk.Button(button_frame, text=browse_text, command=browse_callback)
        browse_button.grid(row=0, column=0, sticky="e")

        if clearable:
            clear_button = ttk.Button(button_frame, text="Clear", command=lambda: variable.set(""))
            clear_button.grid(row=0, column=1, sticky="e", padx=(6, 0))

        return row + 1

    def _browse_encoder(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select tes_encoder executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=str(self.repo_root),
        )
        if selected:
            self.encoder_path_var.set(selected)

    def _browse_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select input MSH",
            filetypes=[("MSH files", "*.msh"), ("All files", "*.*")],
            initialdir=str(self.last_msh_dir),
        )
        if selected:
            self.input_path_var.set(selected)
            self._update_last_msh_dir(selected)
            if not self.output_path_var.get().strip():
                suggested = self.repo_root / "example" / "TES" / (Path(selected).stem + ".json")
                self.output_path_var.set(str(suggested))

    def _browse_outer(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select outer MSH",
            filetypes=[("MSH files", "*.msh"), ("All files", "*.*")],
            initialdir=str(self.last_msh_dir),
        )
        if selected:
            self.outer_path_var.set(selected)
            self._update_last_msh_dir(selected)

    def _browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select output TES",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json",
            initialdir=str(self.last_tes_dir),
        )
        if selected:
            self.output_path_var.set(selected)
            self._update_last_tes_dir(selected)

    def _validate_inputs(self) -> tuple[bool, str]:
        exe_text = self.encoder_path_var.get().strip()
        input_text = self.input_path_var.get().strip()
        outer = self.outer_path_var.get().strip()
        output_text = self.output_path_var.get().strip()

        if not exe_text:
            return False, "Encoder executable path is required."
        if not input_text:
            return False, "Input mesh path is required."
        if not output_text:
            return False, "Output TES path is required."

        exe = Path(exe_text)
        input_mesh = Path(input_text)
        output = Path(output_text)

        if not exe.exists() or not exe.is_file():
            return False, f"Encoder executable not found: {exe}"
        if not input_mesh.exists() or not input_mesh.is_file():
            return False, f"Input mesh not found: {input_mesh}"
        if outer and (not Path(outer).exists() or not Path(outer).is_file()):
            return False, f"Outer mesh not found: {outer}"

        try:
            grid_quality = int(self.grid_quality_var.get().strip())
            if grid_quality <= 0:
                raise ValueError
        except ValueError:
            return False, "Grid quality must be a positive integer."

        try:
            float(self.alpha_var.get().strip())
        except ValueError:
            return False, "Alpha must be a valid floating-point number."

        try:
            float(self.epsilon_var.get().strip())
        except ValueError:
            return False, "Epsilon must be a valid floating-point number."

        output.parent.mkdir(parents=True, exist_ok=True)
        return True, ""

    def _build_command(self) -> list[str]:
        command = [
            self.encoder_path_var.get().strip(),
            "--input",
            self.input_path_var.get().strip(),
            "--output",
            self.output_path_var.get().strip(),
            "--grid-quality",
            self.grid_quality_var.get().strip(),
            "--alpha",
            self.alpha_var.get().strip(),
            "--epsilon",
            self.epsilon_var.get().strip(),
        ]

        outer = self.outer_path_var.get().strip()
        if outer:
            command.extend(["--outer", outer])

        if self.deformable_var.get():
            command.append("--deformable")

        return command

    def _quote_for_preview(self, item: str) -> str:
        if " " in item or "\t" in item:
            return f'"{item}"'
        return item

    def _update_command_preview(self) -> None:
        command = self._build_command()
        self.command_preview_var.set(" ".join(self._quote_for_preview(part) for part in command))

    def _start_encode(self) -> None:
        valid, error = self._validate_inputs()
        if not valid:
            messagebox.showerror("Invalid input", error)
            return

        if self._process is not None:
            messagebox.showwarning("Busy", "An encode process is already running.")
            return

        self._append_log("\n=== Starting encode ===\n")
        self._append_log(self.command_preview_var.get() + "\n\n")

        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        command = self._build_command()
        self._thread = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        self._thread.start()

    def _run_process(self, command: list[str]) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._output_queue.put(f"Failed to launch process: {exc}\n")
            self._output_queue.put("__PROCESS_FINISHED__1")
            return

        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._output_queue.put(line)

        return_code = self._process.wait()
        self._output_queue.put(f"\n=== Process finished with code {return_code} ===\n")
        self._output_queue.put(f"__PROCESS_FINISHED__{return_code}")

    def _stop_encode(self) -> None:
        if self._process is None:
            return

        self._append_log("\nStopping process...\n")
        try:
            self._process.terminate()
        except OSError as exc:
            self._append_log(f"Could not terminate process: {exc}\n")

    def _drain_output_queue(self) -> None:
        while True:
            try:
                item = self._output_queue.get_nowait()
            except queue.Empty:
                break

            if item.startswith("__PROCESS_FINISHED__"):
                self._process = None
                self.run_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
            else:
                self._append_log(item)

        self.root.after(100, self._drain_output_queue)

    def _append_log(self, text: str) -> None:
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)

    def _clear_log(self) -> None:
        self.log_box.delete("1.0", tk.END)


def main() -> int:
    root = tk.Tk()
    app = TesEncoderUi(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
