"""UniInfer CLI — command-line interface using Typer and Rich."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="uniinfer",
    help="UniInfer — Hardware-agnostic AI inference runtime.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def devices() -> None:
    """List all discovered hardware devices."""
    from uniinfer.hal.discovery import devices as discover_devices

    console.print("\n[bold]UniInfer — Hardware Discovery[/bold]\n")

    try:
        found = discover_devices()
    except Exception as exc:
        console.print(f"[red]Error during hardware discovery: {exc}[/red]")
        raise typer.Exit(code=1)

    if not found:
        console.print("[yellow]No devices found.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Available Devices")
    table.add_column("Device", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Type", style="green")
    table.add_column("Total Memory", justify="right", style="magenta")
    table.add_column("Free Memory", justify="right", style="blue")
    table.add_column("Extra", style="dim")

    for dev in found:
        extra_parts = [f"{k}={v}" for k, v in dev.extra.items()]
        if dev.compute_capability:
            extra_parts.insert(0, f"cc={dev.compute_capability[0]}.{dev.compute_capability[1]}")
        extra_str = ", ".join(extra_parts) if extra_parts else ""

        table.add_row(
            dev.device_string,
            dev.name,
            dev.device_type.value.upper(),
            f"{dev.total_memory_gb:.1f} GB",
            f"{dev.free_memory_gb:.1f} GB",
            extra_str,
        )

    console.print(table)
    console.print()


@app.command()
def chat(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local GGUF path"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use (auto, cuda:0, cpu, etc.)"),
    quantization: str = typer.Option("auto", "--quant", "-q", help="Quantization level (auto, f16, q8_0, q4_k_m)"),
    context_length: int = typer.Option(4096, "--ctx", help="Context window size"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Sampling temperature"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per response"),
) -> None:
    """Start an interactive chat session with a model."""
    from uniinfer.engine.engine import Engine

    console.print(f"\n[bold]UniInfer Chat[/bold] — Loading [cyan]{model}[/cyan]...\n")

    try:
        engine = Engine(
            model=model,
            device=device,
            quantization=quantization,
            context_length=context_length,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/red]")
        raise typer.Exit(code=1)

    info = engine.info()
    console.print(f"[green]Model loaded![/green] Device: [cyan]{info['device_name']}[/cyan] ({info['device']})")
    console.print(f"Quantization: [yellow]{info['quantization']}[/yellow] | Context: {info['context_length']} tokens")
    console.print("[dim]Type 'exit' or 'quit' to end the session. Ctrl+C to interrupt.[/dim]\n")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and clearly."},
    ]

    try:
        while True:
            try:
                user_input = console.input("[bold green]You>[/bold green] ")
            except EOFError:
                break

            if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input.strip()})

            console.print("[bold blue]Assistant>[/bold blue] ", end="")
            response_text = ""
            try:
                for chunk in engine.chat_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    console.print(chunk.text, end="", highlight=False)
                    response_text += chunk.text
                console.print()  # newline after response

                messages.append({"role": "assistant", "content": response_text.strip()})
            except KeyboardInterrupt:
                console.print("\n[dim](interrupted)[/dim]")
                if response_text:
                    messages.append({"role": "assistant", "content": response_text.strip()})
            except Exception as exc:
                console.print(f"\n[red]Generation error: {exc}[/red]")

    except KeyboardInterrupt:
        console.print("\n")
    finally:
        engine.close()
        console.print("[dim]Session ended.[/dim]")


@app.command()
def generate(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local GGUF path"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Input prompt"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use"),
    quantization: str = typer.Option("auto", "--quant", "-q", help="Quantization level"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temp", "-t", help="Sampling temperature"),
) -> None:
    """Generate text from a prompt (one-shot)."""
    from uniinfer.engine.engine import Engine

    console.print(f"Loading [cyan]{model}[/cyan]...")

    try:
        engine = Engine(
            model=model,
            device=device,
            quantization=quantization,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/red]")
        raise typer.Exit(code=1)

    try:
        result = engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print(f"\n{result.text}")
        console.print(f"\n[dim]Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion = {result.total_tokens} total[/dim]")
    except Exception as exc:
        console.print(f"[red]Generation failed: {exc}[/red]")
        raise typer.Exit(code=1)
    finally:
        engine.close()


@app.command()
def pull(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID to download"),
    quantization: str = typer.Option("q4_k_m", "--quant", "-q", help="Quantization level to download"),
) -> None:
    """Download and cache a model from HuggingFace."""
    from uniinfer.models.registry import download_model, is_cached

    console.print(f"Pulling [cyan]{model}[/cyan] ({quantization})...")

    if is_cached(model, quantization):
        console.print("[green]Model already cached![/green]")
        return

    try:
        path = download_model(model_id=model, quantization=quantization)
        console.print(f"[green]Model cached at:[/green] {path}")
    except Exception as exc:
        console.print(f"[red]Download failed: {exc}[/red]")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_models() -> None:
    """List all cached models."""
    from uniinfer.models.registry import list_cached

    cached = list_cached()

    if not cached:
        console.print("[yellow]No cached models found.[/yellow]")
        console.print("[dim]Use 'uniinfer pull --model <model_id>' to download a model.[/dim]")
        return

    table = Table(title="Cached Models")
    table.add_column("Model", style="cyan")
    table.add_column("Quantization", style="green")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Source", style="dim")
    table.add_column("Path", style="dim", no_wrap=True)

    for m in cached:
        size_gb = m.file_size / (1024**3)
        size_str = f"{size_gb:.2f} GB" if size_gb >= 1.0 else f"{m.file_size / (1024**2):.1f} MB"
        # Truncate path for display
        path_display = m.gguf_path
        if len(path_display) > 50:
            path_display = "..." + path_display[-47:]

        table.add_row(m.model_id, m.quantization, size_str, m.source, path_display)

    console.print(table)


@app.command()
def bench(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local GGUF path"),
    device: str = typer.Option("auto", "--device", "-d", help="Device to use"),
    quantization: str = typer.Option("auto", "--quant", "-q", help="Quantization level"),
    prompt: str = typer.Option(
        "Explain the theory of general relativity in simple terms.",
        "--prompt",
        "-p",
        help="Benchmark prompt",
    ),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Tokens to generate"),
    runs: int = typer.Option(3, "--runs", "-n", help="Number of benchmark runs"),
) -> None:
    """Run a simple benchmark (tokens/sec)."""
    from uniinfer.engine.engine import Engine

    console.print(f"Benchmarking [cyan]{model}[/cyan] on [green]{device}[/green]...")
    console.print(f"Prompt: [dim]{prompt[:60]}...[/dim]")
    console.print(f"Max tokens: {max_tokens} | Runs: {runs}\n")

    try:
        engine = Engine(
            model=model,
            device=device,
            quantization=quantization,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/red]")
        raise typer.Exit(code=1)

    info = engine.info()
    console.print(f"Device: [cyan]{info['device_name']}[/cyan] | Quantization: [yellow]{info['quantization']}[/yellow]\n")

    results: list[dict[str, float]] = []

    try:
        for i in range(runs):
            console.print(f"  Run {i + 1}/{runs}... ", end="")

            start = time.perf_counter()
            result = engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,  # Greedy for reproducibility
            )
            elapsed = time.perf_counter() - start

            tokens = result.completion_tokens
            tps = tokens / elapsed if elapsed > 0 else 0

            results.append({
                "tokens": tokens,
                "elapsed": elapsed,
                "tps": tps,
            })

            console.print(f"[green]{tokens} tokens[/green] in {elapsed:.2f}s = [bold]{tps:.1f} tok/s[/bold]")

    except Exception as exc:
        console.print(f"\n[red]Benchmark failed: {exc}[/red]")
        raise typer.Exit(code=1)
    finally:
        engine.close()

    # Summary
    if results:
        avg_tps = sum(r["tps"] for r in results) / len(results)
        max_tps = max(r["tps"] for r in results)
        console.print(f"\n[bold]Average: {avg_tps:.1f} tok/s | Peak: {max_tps:.1f} tok/s[/bold]")


if __name__ == "__main__":
    app()
