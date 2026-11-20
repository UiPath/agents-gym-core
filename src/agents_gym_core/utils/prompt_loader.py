"""Loader for prompt files."""

from agents_gym_core.utils.directory_provider import DirectoryProvider


class PromptLoader:
    """Loader for prompt files."""

    @staticmethod
    def load_prompt_from_file(
        prompt_value: str, agent_name: str, directory_provider: DirectoryProvider
    ) -> str:
        """Return ``prompt_value``, or the stripped text of the file it references.

        A reference such as ``core::prompts/system_prompt.md`` is resolved against the agent's
        directory by ``directory_provider``; any other string is the prompt itself.

        Raises:
            FileNotFoundError: If the referenced file does not exist.
        """
        if not isinstance(prompt_value, str):
            raise Exception("The argument: 'prompt_value' should be a string.")

        if not directory_provider.check_for_placeholder(prompt_value):
            return prompt_value

        prompt_file = directory_provider.solve_path(agent_name, prompt_value)
        if prompt_file.exists():
            return prompt_file.read_text().strip()

        raise FileNotFoundError(f"Prompt file not found: {prompt_value}. Tried: {prompt_file}.")
