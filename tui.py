import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.prompt import Confirm


def list_files(directory: str) -> dict[str, str]:
    py_dict = {}
    for path in Path(directory).rglob("*.py"):
        py_dict[path.stem] = str(path)
    return py_dict


@dataclass
class Module:
    name: str
    preprocessing: dict[str, str]
    edit: dict[str, str]
    train: str | None = None
    test: str | None = None
    submodules: dict[str, "Module"] = field(default_factory=dict)


extractor = Module(
    name="extractor",
    train="models/extractor/train.py",
    test="models/extractor/test.py",
    preprocessing=list_files("models/extractor/preprocessing"),
    edit=list_files("models/extractor"),
)

transformer = Module(
    name="transformer",
    train="models/transformer/train.py",
    test="models/transformer/test.py",
    preprocessing=list_files("models/transformer/preprocessing"),
    edit=list_files("models/transformer"),
)

generator = Module(
    name="generator",
    preprocessing={"spec_gen": "models/generator/preprocessing/spec_gen.py"},
    edit=list_files("models/generator"),
)

sign2speech = Module(
    name="sign2speech",
    train="models/train.py",
    test="models/test.py",
    preprocessing={},
    submodules={module.name: module for module in [extractor, transformer, generator]},
    edit={p.stem: str(p) for p in Path("models").iterdir() if p.suffix == ".py"},
)


class REPL:
    def __init__(self) -> None:
        self.context = [sign2speech]
        self.console = Console()
        self.session = PromptSession(
            history=FileHistory(".tui_history"),
            completer=NestedCompleter.from_nested_dict(self.commands),
        )
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = os.getcwd()

    @property
    def commands(self) -> dict[str, Any]:
        module = self.context[-1]
        commands = {
            "train": None,
            "test": None,
            "preprocess": {v: None for v in module.preprocessing.keys()},
            "edit": {v: None for v in module.edit.keys()} | {"config": None},
            "back": None,
            "quit": None,
            "exit": None,
        }
        commands["bg"] = commands.copy()
        return commands | {v: None for v in module.submodules.keys()}

    def handle_edit(self, args: list[str]) -> list[str]:
        if len(args) == 0:
            self.console.print("Enter file name you want to edit.", style="yellow")
            return []

        if args[0] in self.context[-1].edit:
            return ["nvim", self.context[-1].edit[args[0]]]

        if args[0] in self.context[-1].submodules and len(args) > 1:
            module = self.context[-1].submodules[args[0]]
            if args[1] in module.edit:
                return ["nvim", module.edit[args[1]]]

        if args[0] == "config":
            return ["nvim", "config.yaml"]

        self.console.print("No such file exists.", style="red")
        return []

    def handle_train(self) -> list[str]:
        if not self.context[-1].train:
            self.console.print("No train script found.", style="red")
            return []
        return ["uv", "run", self.context[-1].train]

    def handle_test(self, args) -> list[str]:
        if not self.context[-1].test:
            self.console.print("No test script found.", style="red")
            return []
        return ["uv", "run", self.context[-1].test] + args

    def handle_preprocess(self, args: list[str]) -> list[str]:
        if len(args) == 0:
            self.console.print("Enter preprocessing script name.", style="yellow")
            return []

        if args[0] in self.context[-1].preprocessing:
            return ["uv", "run", self.context[-1].preprocessing[args[0]]]

        self.console.print("No such preprocessing script found.", style="red")
        return []

    def handle_back(self) -> list[str]:
        if len(self.context) > 1:
            self.context.pop()

        self.session.completer = NestedCompleter.from_nested_dict(self.commands)
        return []

    def handle_forward(self, module: str) -> list[str]:
        self.context.append(self.context[-1].submodules[module])
        self.session.completer = NestedCompleter.from_nested_dict(self.commands)
        return []

    def conf_command(self, command_str: list[str]) -> bool:
        if len(command_str) == 0:
            return False

        cmd_display = " ".join(command_str)
        self.console.print(f"Running: [bold cyan]{cmd_display}[/bold cyan]")
        return Confirm.ask("Continue?", console=self.console, default=False)

    def execute_command(self, command_str: list[str], background: bool = False) -> None:
        try:
            if background:
                subprocess.run(
                    " ".join(command_str), shell=True, check=True, env=self.env
                )
            else:
                subprocess.run(command_str, check=True, env=self.env)
        except subprocess.CalledProcessError as e:
            self.console.print(
                f"Command failed with exit code [bold red]{e.returncode}[/bold red]",
                style="red",
            )
        except FileNotFoundError:
            self.console.print(
                f"Command not found: [bold red]{command_str[0]}[/bold red]",
                style="red",
            )

    def process_input(self, inp: str) -> list[str] | None:
        if len(inp) == 0:
            return []

        command, *args = inp.split()

        if command == "bg":
            command_str = self.process_input(" ".join(args))
            if command_str is None or len(command_str) == 0:
                return []
            return ["nohup"] + command_str + [">", "out.log", "2>&1", "&"]

        if command == "quit" or command == "exit":
            return None

        if command == "edit":
            return self.handle_edit(args)

        if command == "train":
            command_str = self.handle_train()
            if not self.conf_command(command_str):
                return []
            return command_str

        if command == "test":
            command_str = self.handle_test(args)
            if not self.conf_command(command_str):
                return []
            return command_str

        if command == "preprocess":
            command_str = self.handle_preprocess(args)
            if not self.conf_command(command_str):
                return []
            return command_str

        if command == "back":
            return self.handle_back()

        if command in self.context[-1].submodules:
            return self.handle_forward(command)

        else:
            self.console.print("Command not found.", style="red")
            return []

    def run(self) -> None:
        self.console.print("[bold blue]Sign2Speech ML Pipeline REPL[/bold blue]")
        self.console.print("Type commands or use tab completion. Use 'quit' to exit.\n")

        while True:
            try:
                prompt_text = HTML(
                    f"<ansigreen>({self.context[-1].name})</ansigreen> > "
                )
                inp = self.session.prompt(prompt_text)
                result = self.process_input(inp)

                if result is None:
                    self.console.print("Goodbye! 👋", style="bold blue")
                    break
                elif len(result) == 0:
                    continue
                elif result[0] == "nohup":
                    self.execute_command(result, background=True)
                else:
                    self.execute_command(result)

            except KeyboardInterrupt:
                continue
            except EOFError:
                self.console.print("\nGoodbye! 👋", style="bold blue")
                break
            except Exception as e:
                self.console.print(f"Error: [bold red]{e}[/bold red]", style="red")
                continue


if __name__ == "__main__":
    repl = REPL()
    repl.run()
