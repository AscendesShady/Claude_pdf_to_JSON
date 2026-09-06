"""Tkinter desktop app: queue PDFs, configure the pipeline, run it, watch live progress."""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import sv_ttk

import gpu_monitor
import ollama_client
import pipeline
from config import AVAILABLE_MODELS_FALLBACK, PipelineConfig
from progress import ProgressEvent

GPU_POLL_INTERVAL_S = 1.0

logger = logging.getLogger(__name__)


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue"):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.log_queue.put(("log", msg))


class StatTile(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str, value_color: str | None = None):
        super().__init__(parent, padding=(12, 8), relief="groove", borderwidth=1)
        self.value_var = tk.StringVar(value="0")
        ttk.Label(self, text=title, font=("Segoe UI", 9)).pack(anchor="w")
        value_label = ttk.Label(self, textvariable=self.value_var, font=("Segoe UI", 18, "bold"))
        if value_color:
            value_label.configure(foreground=value_color)
        value_label.pack(anchor="w")

    def set(self, value: object) -> None:
        self.value_var.set(str(value))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        sv_ttk.set_theme("dark")

        self.title("PDF to QLoRA JSONL")
        self.geometry("980x760")
        self.minsize(820, 620)

        self.pdf_paths: list[str] = []
        self.msg_queue: "queue.Queue" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self._active_handler: logging.Handler | None = None
        self.run_start_time: float | None = None
        self.last_result: "pipeline.PipelineResult | None" = None
        self.ollama_host = PipelineConfig().ollama_host

        self._gpu_stop = threading.Event()

        self._build_widgets()
        self._populate_models()
        self._start_gpu_monitor()
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_gpu_monitor(self) -> None:
        """Poll VRAM on its own thread so the reading stays live regardless of what the
        pipeline is doing (a single chunk can take a minute on a large model)."""
        self.tile_vram.set("...")

        def worker() -> None:
            while not self._gpu_stop.is_set():
                self.msg_queue.put(("gpu", gpu_monitor.read_vram()))
                self._gpu_stop.wait(GPU_POLL_INTERVAL_S)

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        self._gpu_stop.set()
        self.destroy()

    # ---------- UI construction ----------

    def _build_widgets(self) -> None:
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(top_bar, text="PDF → QLoRA JSONL", font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Button(top_bar, text="Toggle Theme", command=self._toggle_theme).pack(side="right")

        file_frame = ttk.LabelFrame(self, text="PDF Queue")
        file_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.file_listbox = tk.Listbox(file_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(side="left", padx=5, pady=5)
        ttk.Button(btn_frame, text="Add PDFs", command=self._add_pdfs).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(fill="x", pady=2)

        cfg_frame = ttk.LabelFrame(self, text="Configuration")
        cfg_frame.pack(fill="x", padx=10, pady=5)
        cfg_frame.columnconfigure(1, weight=1)
        cfg_frame.columnconfigure(3, weight=1)

        r = 0
        ttk.Label(cfg_frame, text="Model:").grid(row=r, column=0, sticky="w", padx=5, pady=3)
        self.model_var = tk.StringVar(value="gemma4:12b")
        self.model_combo = ttk.Combobox(
            cfg_frame, textvariable=self.model_var, values=AVAILABLE_MODELS_FALLBACK,
            width=22, state="readonly",
        )
        self.model_combo.grid(row=r, column=1, sticky="w", padx=5, pady=3)

        ttk.Label(cfg_frame, text="Temperature:").grid(row=r, column=2, sticky="w", padx=5, pady=3)
        self.temp_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(cfg_frame, from_=0.0, to=0.3, increment=0.05, textvariable=self.temp_var, width=6).grid(
            row=r, column=3, sticky="w", padx=5, pady=3
        )
        r += 1

        # Own row so the status text always has the full frame width and never competes
        # with other controls for space (that competition was clipping it before).
        status_row = ttk.Frame(cfg_frame)
        status_row.grid(row=r, column=0, columnspan=4, sticky="ew", padx=5, pady=(0, 3))
        ttk.Button(status_row, text="Refresh", width=8, command=self._populate_models).pack(side="left")
        self.ollama_status_var = tk.StringVar(value="Checking Ollama...")
        self.ollama_status_label = ttk.Label(status_row, textvariable=self.ollama_status_var)
        self.ollama_status_label.pack(side="left", padx=8, fill="x", expand=True)
        status_row.bind(
            "<Configure>",
            lambda e: self.ollama_status_label.configure(wraplength=max(e.width - 100, 100)),
        )
        r += 1

        ttk.Label(cfg_frame, text="Chunk tokens:").grid(row=r, column=0, sticky="w", padx=5, pady=3)
        self.chunk_tokens_var = tk.IntVar(value=600)
        ttk.Spinbox(cfg_frame, from_=200, to=1200, increment=50, textvariable=self.chunk_tokens_var, width=8).grid(
            row=r, column=1, sticky="w", padx=5, pady=3
        )

        ttk.Label(cfg_frame, text="Overlap %:").grid(row=r, column=2, sticky="w", padx=5, pady=3)
        self.overlap_var = tk.DoubleVar(value=0.15)
        ttk.Spinbox(cfg_frame, from_=0.05, to=0.3, increment=0.05, textvariable=self.overlap_var, width=6).grid(
            row=r, column=3, sticky="w", padx=5, pady=3
        )
        r += 1

        ttk.Label(cfg_frame, text="Min chunk tokens:").grid(row=r, column=0, sticky="w", padx=5, pady=3)
        self.min_tokens_var = tk.IntVar(value=40)
        ttk.Spinbox(cfg_frame, from_=10, to=200, increment=10, textvariable=self.min_tokens_var, width=8).grid(
            row=r, column=1, sticky="w", padx=5, pady=3
        )

        ttk.Label(cfg_frame, text="Variants/chunk:").grid(row=r, column=2, sticky="w", padx=5, pady=3)
        self.variants_var = tk.IntVar(value=1)
        ttk.Spinbox(cfg_frame, from_=0, to=5, textvariable=self.variants_var, width=6).grid(
            row=r, column=3, sticky="w", padx=5, pady=3
        )
        r += 1

        ttk.Label(cfg_frame, text="Train/Val/Test ratio:").grid(row=r, column=0, sticky="w", padx=5, pady=3)
        ratio_frame = ttk.Frame(cfg_frame)
        ratio_frame.grid(row=r, column=1, columnspan=3, sticky="w")
        self.train_ratio_var = tk.DoubleVar(value=0.8)
        self.val_ratio_var = tk.DoubleVar(value=0.1)
        self.test_ratio_var = tk.DoubleVar(value=0.1)
        ttk.Entry(ratio_frame, textvariable=self.train_ratio_var, width=5).pack(side="left", padx=2)
        ttk.Entry(ratio_frame, textvariable=self.val_ratio_var, width=5).pack(side="left", padx=2)
        ttk.Entry(ratio_frame, textvariable=self.test_ratio_var, width=5).pack(side="left", padx=2)
        r += 1

        ttk.Label(cfg_frame, text="Output root:").grid(row=r, column=0, sticky="w", padx=5, pady=3)
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "output"))
        ttk.Entry(cfg_frame, textvariable=self.output_dir_var).grid(
            row=r, column=1, columnspan=2, sticky="ew", padx=5, pady=3
        )
        ttk.Button(cfg_frame, text="Browse", command=self._browse_output).grid(
            row=r, column=3, sticky="w", padx=5, pady=3
        )

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(ctrl_frame, text="Start", style="Accent.TButton", command=self._start)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.open_review_btn = ttk.Button(
            ctrl_frame, text="Open Review Sheet", command=self._open_review_sheet, state="disabled"
        )
        self.open_review_btn.pack(side="left", padx=5)
        self.open_folder_btn = ttk.Button(
            ctrl_frame, text="Open Output Folder", command=self._open_output_folder, state="disabled"
        )
        self.open_folder_btn.pack(side="left", padx=5)

        # --- Live stats dashboard ---
        stats_frame = ttk.LabelFrame(self, text="Live Progress")
        stats_frame.pack(fill="x", padx=10, pady=5)

        # Two rows of four - eight tiles on one row gets cramped at narrow window widths.
        tiles_row1 = ttk.Frame(stats_frame)
        tiles_row1.pack(fill="x", padx=5, pady=(5, 2))
        self.tile_pages = StatTile(tiles_row1, "Pages Processed")
        self.tile_chunks = StatTile(tiles_row1, "Chunks Created")
        self.tile_examples = StatTile(tiles_row1, "Examples Generated")
        self.tile_elapsed = StatTile(tiles_row1, "Elapsed")

        tiles_row2 = ttk.Frame(stats_frame)
        tiles_row2.pack(fill="x", padx=5, pady=(2, 5))
        self.tile_qc_passed = StatTile(tiles_row2, "QC Passed", value_color="#3fb950")
        self.tile_qc_rejected = StatTile(tiles_row2, "QC Rejected", value_color="#f85149")
        self.tile_tokens = StatTile(tiles_row2, "Tokens Used")
        self.tile_vram = StatTile(tiles_row2, "VRAM Used")

        for tile in (
            self.tile_pages, self.tile_chunks, self.tile_examples, self.tile_elapsed,
            self.tile_qc_passed, self.tile_qc_rejected, self.tile_tokens, self.tile_vram,
        ):
            tile.pack(side="left", fill="both", expand=True, padx=4)

        stage_frame = ttk.Frame(stats_frame)
        stage_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.stage_var = tk.StringVar(value="Idle")
        self.task_var = tk.StringVar(value="")
        ttk.Label(stage_frame, text="Stage:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(stage_frame, textvariable=self.stage_var).pack(side="left", padx=(4, 20))
        ttk.Label(stage_frame, text="Task:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(stage_frame, textvariable=self.task_var).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(stats_frame, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=(0, 5))

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_text = tk.Text(log_frame, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, side="left")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _toggle_theme(self) -> None:
        sv_ttk.set_theme("light" if sv_ttk.get_theme() == "dark" else "dark")

    def _populate_models(self) -> None:
        if not ollama_client.is_reachable(self.ollama_host):
            self.model_combo["values"] = AVAILABLE_MODELS_FALLBACK
            self.ollama_status_var.set("Ollama: not reachable - is it installed and running?")
            return

        models = ollama_client.list_models(self.ollama_host)
        if models:
            self.model_combo["values"] = models
            if self.model_var.get() not in models:
                self.model_var.set(models[0])
            self.ollama_status_var.set(f"Ollama: {len(models)} model(s) available")
        else:
            self.model_combo["values"] = AVAILABLE_MODELS_FALLBACK
            self.ollama_status_var.set("Ollama: running, but no models pulled yet")

    # ---------- File queue ----------

    def _add_pdfs(self) -> None:
        paths = filedialog.askopenfilenames(title="Select PDFs", filetypes=[("PDF files", "*.pdf")])
        for p in paths:
            if p not in self.pdf_paths:
                self.pdf_paths.append(p)
                self.file_listbox.insert(tk.END, Path(p).name)

    def _remove_selected(self) -> None:
        selected = list(self.file_listbox.curselection())
        for idx in reversed(selected):
            self.file_listbox.delete(idx)
            del self.pdf_paths[idx]

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir_var.set(d)

    # ---------- Run control ----------

    def _build_config(self) -> PipelineConfig:
        return PipelineConfig(
            ollama_host=self.ollama_host,
            model_name=self.model_var.get().strip(),
            temperature=float(self.temp_var.get()),
            target_chunk_tokens=int(self.chunk_tokens_var.get()),
            chunk_overlap_pct=float(self.overlap_var.get()),
            min_chunk_tokens=int(self.min_tokens_var.get()),
            variants_per_chunk=int(self.variants_var.get()),
            train_ratio=float(self.train_ratio_var.get()),
            val_ratio=float(self.val_ratio_var.get()),
            test_ratio=float(self.test_ratio_var.get()),
            output_dir=self.output_dir_var.get(),
        )

    def _reset_dashboard(self) -> None:
        # tile_vram is deliberately excluded - it shows live system state, not run progress.
        for tile in (
            self.tile_pages, self.tile_chunks, self.tile_examples, self.tile_elapsed,
            self.tile_qc_passed, self.tile_qc_rejected, self.tile_tokens,
        ):
            tile.set(0)
        self.stage_var.set("Starting...")
        self.task_var.set("")
        self.progress["value"] = 0

    def _start(self) -> None:
        if not self.pdf_paths:
            messagebox.showwarning("No PDFs", "Add at least one PDF before starting.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return

        config = self._build_config()

        if not ollama_client.is_reachable(config.ollama_host):
            messagebox.showerror(
                "Ollama not reachable",
                f"Could not reach Ollama at {config.ollama_host}.\n\n"
                "Install it from https://ollama.com and make sure it's running, then try again.",
            )
            return

        pdf_paths = list(self.pdf_paths)
        self._launch_run(pdf_paths, config)

    def _launch_run(self, pdf_paths: list[str], config: PipelineConfig) -> None:
        self.stop_event = threading.Event()
        self._set_log_text("")
        self._reset_dashboard()
        self.run_start_time = time.time()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.open_review_btn.configure(state="disabled")
        self.open_folder_btn.configure(state="disabled")

        handler = QueueLogHandler(self.msg_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        self._active_handler = handler

        stop_event = self.stop_event

        def progress_cb(event: ProgressEvent) -> None:
            self.msg_queue.put(("progress", event))

        def worker() -> None:
            try:
                result = pipeline.run_pipeline(
                    pdf_paths, config, progress_cb=progress_cb, stop_event=stop_event
                )
                self.msg_queue.put(("done", result))
            except Exception as exc:  # surface any failure to the GUI instead of dying silently
                logger.exception("Pipeline failed")
                self.msg_queue.put(("error", str(exc)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.stage_var.set("Stopping...")

    # ---------- Queue polling ----------

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "progress":
                    self._apply_progress(item[1])
                elif kind == "gpu":
                    self._apply_gpu(item[1])
                elif kind == "done":
                    self._on_finished(item[1])
                elif kind == "error":
                    self._on_error(item[1])
        except queue.Empty:
            pass

        if self.run_start_time is not None and self.worker_thread is not None and self.worker_thread.is_alive():
            elapsed = int(time.time() - self.run_start_time)
            self.tile_elapsed.set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

        self.after(100, self._poll_queue)

    def _apply_progress(self, event: ProgressEvent) -> None:
        self.stage_var.set(event.stage)
        self.task_var.set(event.current_task)
        self.tile_pages.set(f"{event.pages_done}/{event.pages_total}")
        self.tile_chunks.set(event.chunks_created)
        self.tile_examples.set(event.examples_generated)
        self.tile_qc_passed.set(event.qc_passed)
        self.tile_qc_rejected.set(event.qc_rejected)
        self.tile_tokens.set(f"{event.tokens_used:,}")
        if event.total:
            self.progress["maximum"] = event.total
            self.progress["value"] = event.done

    def _apply_gpu(self, reading: tuple[float, float] | None) -> None:
        if reading is None:
            self.tile_vram.set("n/a")
            return
        used_mb, total_mb = reading
        self.tile_vram.set(f"{used_mb / 1024:.1f} / {total_mb / 1024:.1f} GB")

    def _append_log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_log_text(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        if text:
            self.log_text.insert(tk.END, text)
        self.log_text.configure(state="disabled")

    def _cleanup_handler(self) -> None:
        if self._active_handler is not None:
            logging.getLogger().removeHandler(self._active_handler)
            self._active_handler = None

    def _on_finished(self, result: "pipeline.PipelineResult") -> None:
        self._cleanup_handler()
        self.run_start_time = None
        self.last_result = result
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.open_review_btn.configure(state="normal")
        self.open_folder_btn.configure(state="normal")
        self.stage_var.set("Done")
        summary = (
            f"Pages processed: {result.total_pages}\n"
            f"Chunks created: {result.total_chunks}\n"
            f"Examples generated: {result.total_examples_generated}\n"
            f"QC passed: {result.total_qc_passed}\n"
            f"QC rejected: {result.total_qc_rejected}\n"
            f"Tokens used: {result.total_tokens_used:,}\n"
            f"Split counts: {result.split_counts}\n\n"
            f"Run folder: {result.output_dir}\n"
            f"Review workbook: {Path(result.review_workbook_path).name}\n\n"
            f"Open the review workbook to mark Keep/Reject/Needs Fix on each generated "
            f"pair before trusting the JSONL for training."
        )
        messagebox.showinfo("Run complete", summary)

    def _open_review_sheet(self) -> None:
        if self.last_result is None:
            return
        try:
            os.startfile(self.last_result.review_workbook_path)  # noqa: S606 (Windows-only app)
        except OSError as exc:
            messagebox.showerror("Could not open file", str(exc))

    def _open_output_folder(self) -> None:
        if self.last_result is None:
            return
        try:
            os.startfile(self.last_result.output_dir)  # noqa: S606 (Windows-only app)
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def _on_error(self, message: str) -> None:
        self._cleanup_handler()
        self.run_start_time = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.stage_var.set("Error")
        messagebox.showerror("Pipeline error", message)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
