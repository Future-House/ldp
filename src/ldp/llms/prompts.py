"""This module provides utility functions for appending and prepending system messages."""

from collections.abc import Collection, Iterable

from aviary.core import Message


def append_to_messages(messages: list[Message], new_message: Message) -> list[Message]:
    """Appends a message to a list of messages, returning that in-place modified list.

    Examples:
        >>> messages = [Message(content="Hello")]
        >>> modified_messages = append_to_messages(messages, Message(content="New"))
        >>> modified_messages
        [Message(role='user', content='Hello'), Message(role='user', content='New')]
        >>> id(messages) == id(modified_messages)
        True
    """
    messages.append(new_message)
    return messages


def append_to_sys(user_content: str, sys_content: str | None = None) -> list[Message]:
    """Appends a user message to a list of messages, optionally including a system message.

    Args:
        user_content: The content of the user message.
        sys_content: Optional content for the system message. Defaults to None.

    Returns:
        A list of messages including the optional system message and the user message.

    Examples:
        >>> append_to_sys("Hello, world!")
        [Message(role='user', content='Hello, world!')]

        >>> append_to_sys("Hello, world!", "System initialized.")
        [Message(role='system', content='System initialized.'), Message(role='user', content='Hello, world!')]
    """
    sys = [Message(role="system", content=sys_content)] if sys_content else []
    return [*sys, Message(content=user_content)]


def prepend_sys(messages: Collection, sys_content: str) -> list[Message]:
    """Prepends a system message to a list of messages.

    Args:
        messages: The list of existing messages.
        sys_content: The content of the system message to be prepended.

    Returns:
        A new list of messages with the system message prepended.

    Examples:
        >>> messages = [Message(role="user", content="Hello!")]
        >>> prepend_sys(messages, "System initialized.")
        [Message(role='system', content='System initialized.'), Message(role='user', content='Hello!')]
    """
    return [Message(role="system", content=sys_content), *messages]


def indent_xml(xml_string, indent_size=2):
    output = []
    indent_level = 0

    # Split the input XML into parts by tags
    parts = xml_string.replace(">", ">\n").replace("<", "\n<").split("\n")
    parts = [part for part in parts if part.strip()]  # Remove empty parts

    for part in parts:
        if part.startswith("</"):
            # Closing tag, decrease indent
            indent_level -= indent_size
            output.append(" " * indent_level + part)
        elif part.startswith("<") and not part.endswith("/>") and ">" in part:
            # Opening tag, maintain then increase indent
            output.append(" " * indent_level + part)
            indent_level += indent_size
        elif part.endswith("/>"):
            # Self-closing tag, just append
            output.append(" " * indent_level + part)
        else:
            # Text or other data, maintain current indent
            # Handle multiple lines within text nodes
            text_lines = part.split("\n")
            output.extend([
                " " * indent_level + line.strip() for line in text_lines if line.strip()
            ])

    return "\n".join(output)


def prepend_sys_and_append_sys(
    messages: Iterable[Message], initial_sys_content: str, final_sys_content: str
) -> list[Message]:
    return [
        Message(role="system", content=initial_sys_content),
        *messages,
        Message(role="system", content=final_sys_content),
    ]
