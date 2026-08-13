from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional, List
import json
import re
import uuid
class HistoryNode:
    def __init__(self, parent: Optional['HistoryNode'] = None):
        self.id = str(uuid.uuid4()) 
        self.action: Optional[str] = None
        self.element: Optional[str] = None
        self.intent: Optional[str] = None
        self.url: Optional[str] = None
        self.axtree: Optional[str] = None
        self.subtask: Optional[str] = None
        self.parent = parent
        self.children: List['HistoryNode'] = []
        self.is_error = False
        self.error_info = None

    def add_child(self, child: 'HistoryNode'):
        self.children.append(child)

    def __repr__(self):
        return f"HistoryNode(id={self.id[:6]}, action={self.action}, url={self.url})"

    def get_action(self):
        return self.action

    def get_element(self):
        return self.element

    def get_url(self):
        return self.url

    def get_intent(self):
        return self.intent

    def get_axtree(self):
        return self.axtree

    def get_subtask(self):
        return self.subtask

    def get_error_info(self):
        return self.error_info

    def get_error_state(self):
        return self.is_error

    def set_action(self, action):
        self.action = action

    def set_element(self, element):
        self.element = element

    def set_intent(self, intent):
        self.intent = intent

    def set_url(self, url):
        self.url = url

    def set_axtree(self, axtree):
        self.axtree = axtree

    def set_subtask(self, subtask):
        self.subtask = subtask

    def set_error_info(self, error_info):
        self.error_info = error_info

    def set_as_error(self):
        self.is_error = True

class HistoryTree:
    def __init__(self):
        self.root = HistoryNode()
        self.current = self.root
        self.nodes = {self.root.id: self.root}

    def add_empty_node(self) -> HistoryNode:

        new_node = HistoryNode(parent=self.current)
        self.current.add_child(new_node)
        self.current = new_node
        self.nodes[new_node.id] = new_node
        return new_node

    def backtrack(self):
        if self.current.parent:
            self.current = self.current.parent
        else:
            raise Exception("Has traced back to the root node.")

    def get_current_path(self) -> List[HistoryNode]:
        path = []
        node = self.current
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_current_node(self):
        return self.current

    def get_history(self) -> str:
        history = []
        node = self.current
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"action: {node.get_action()}, manipulated element: {node.element}, action summary: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        idx = 1
        for h in history:
            h = f"Step {idx}. {h}"
            idx += 1
        history = "\n".join(history)
        return history
    
    def get_n_history(self, n) -> str:
        history = []
        node = self.current
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"action: {node.get_action()}, manipulated element: {node.element}, action summary: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        idx = 1
        for h in history:
            h = f"Step {idx}. {h}"
            idx += 1
        history = history[-n:]
        history = "\n".join(history)
        return history
    def get_any_node_history(self, node) -> str:
        history = []
        if node.action is None:
            return "No history, this is your first step."
        while node:
            if node.action is None:
                break
            history.append(f"action: {node.get_action()}, manipulated element: {node.element}, action intent: {node.get_intent()}\n")
            node = node.parent

        history = list(reversed(history))
        history = "\n".join(history)
        return history

    def parent_history(self) -> str:
        parent_node = self.current.parent
        if parent_node.get_action() == None:
            return "No history, this is the first step."
        else:
            return self.get_any_node_history(parent_node)

    def get_error_descendants(self):

        errors = []
        for child in self.current.children:
            if child.is_error:
                errors.append(child.error_info)
        if len(errors) == 0:
            return "This is your first attempt."
        else:
            errors = "\n".join(errors)
        return errors