from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

import argparse
import locale
import math
import time

import code  # noqa
import readline  # noqa
import rlcompleter  # noqa


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


class Interview:
    # Data Structures: Binary Tree
    class TreeNode:
        def __init__(self, value=0, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right

    # Data Structure: Stack
    class Stack:
        def __init__(self):
            self.items = []

        def push(self, item):
            self.items.append(item)

        def pop(self):
            if not self.is_empty():
                return self.items.pop()
            return None

        def peek(self):
            if not self.is_empty():
                return self.items[-1]
            return None

        def is_empty(self):
            return len(self.items) == 0

        def size(self):
            return len(self.items)

    # Data Structure: Queue
    class Queue:
        def __init__(self):
            self.items = []

        def enqueue(self, item):
            self.items.append(item)

        def dequeue(self):
            if not self.is_empty():
                return self.items.pop(0)
            return None

        def is_empty(self):
            return len(self.items) == 0

        def size(self):
            return len(self.items)

    # Data Structure: Linked List
    class ListNode:
        def __init__(self, value=0, next=None):
            self.value = value
            self.next = next

    def __init__(self):
        pass

    def _factorial_recursive(self, n):
        if n == 0:
            return 1
        return n * self._factorial_recursive(n - 1)

    def _factorial_divide_and_conquer(self, low, high):
        if low > high:
            return 1
        if low == high:
            return low
        mid = (low + high) // 2
        return self._factorial_divide_and_conquer(
            low, mid
        ) * self._factorial_divide_and_conquer(mid + 1, high)

    # Recursive Factorial with Timing
    def factorial_recursive(self, n):
        start_time = time.time()  # Start timing
        result = self._factorial_recursive(n)  # Calculate factorial
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}  Elapsed time: {elapsed_time:.6f}"

    # Iterative Factorial with Timing
    def factorial_iterative(self, n):
        start_time = time.time()  # Start timing
        result = 1
        for i in range(1, n + 1):
            result *= i
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}  Elapsed time: {elapsed_time:.6f}"

    # Divide and Conquer Factorial with Timing
    def factorial_divide_and_conquer(self, n):
        start_time = time.time()  # Start timing
        result = self._factorial_divide_and_conquer(1, n)  # Calculate factorial
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}  Elapsed time: {elapsed_time:.6f}"

    # Built-in Factorial with Timing
    def factorial_builtin(self, n):
        start_time = time.time()  # Start timing
        result = math.factorial(n)  # Calculate factorial using built-in
        end_time = time.time()  # End timing

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print complexity and runtime
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}  Elapsed time: {elapsed_time:.6f}"

    # Recursion: Fibonacci
    def fibonacci_recursive(self, n):
        if n <= 1:
            return n
        return self.fibonacci_recursive(n - 1) + self.fibonacci_recursive(n - 2)

    # Iteration: Fibonacci
    def fibonacci_iterative(self, n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    # Searching: Linear Search
    def linear_search(self, arr, target):
        for i, value in enumerate(arr):
            if value == target:
                return i
        return -1

    # Searching: Binary Search
    def binary_search(self, arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    # Sorting: Bubble Sort
    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    # Sorting: Merge Sort
    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            self.merge_sort(left_half)
            self.merge_sort(right_half)

            i = j = k = 0

            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1
        return arr

    def insert_linked_list(self, head, value):
        new_node = self.ListNode(value)
        if not head:
            return new_node
        current = head
        while current.next:
            current = current.next
        current.next = new_node
        return head

    def print_linked_list(self, head):
        current = head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")

    def inorder_traversal(self, root):
        return (
            self.inorder_traversal(root.left)
            + [root.value]
            + self.inorder_traversal(root.right)
            if root
            else []
        )

    def preorder_traversal(self, root):
        return (
            [root.value]
            + self.preorder_traversal(root.left)
            + self.preorder_traversal(root.right)
            if root
            else []
        )

    def postorder_traversal(self, root):
        return (
            self.postorder_traversal(root.left)
            + self.postorder_traversal(root.right)
            + [root.value]
            if root
            else []
        )

    # Graph Algorithms: Depth-First Search
    def dfs(self, graph, start):
        visited, stack = set(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                stack.extend(set(graph[vertex]) - visited)
        return visited

    # Graph Algorithms: Breadth-First Search
    def bfs(self, graph, start):
        visited, queue = set(), [start]
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend(set(graph[vertex]) - visited)
        return visited


def setup_readline(local):
    # Enable tab completion
    readline.parse_and_bind("tab: complete")
    # Optionally, you can set the completer function manually
    readline.set_completer(rlcompleter.Completer(local).complete)


def main():
    console = Console()
    interview = Interview()

    parser = argparse.ArgumentParser(description="Programming Interview Questions")

    parser.add_argument("-f", "--factorial", type=int, help="Factorial number")
    parser.add_argument("--fibonacci", type=int, help="Fibonacci number")
    parser.add_argument("--search", action="store_true", help="Search examples")
    parser.add_argument("--sort", action="store_true", help="Search examples")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )

    args = parser.parse_args()

    if args.factorial:
        # Factorial examples
        console.rule("Factorial Examples")
        rprint(
            Panel(
                "[bold cyan]Recursive Factorial - Time Complexity: O(n)[/bold cyan]\n"
                + str(interview.factorial_recursive(args.factorial)),
                title="Factorial Recursive",
            )
        )
        rprint(
            Panel(
                "[bold cyan]Iterative Factorial - Time Complexity: O(n)[/bold cyan]\n"
                + str(interview.factorial_iterative(args.factorial)),
                title="Factorial Iterative",
            )
        )
        rprint(
            Panel(
                "[bold cyan]Built-in Factorial - Time Complexity: O(n)[/bold cyan]\n"
                + str(interview.factorial_builtin(args.factorial)),
                title="Factorial Built-in",
            )
        )
        rprint(
            Panel(
                "[bold cyan]Divide and Conquer Factorial - Time Complexity: O(n log n)[/bold cyan]\n"
                + str(interview.factorial_divide_and_conquer(args.factorial)),
                title="Factorial Divide and Conquer",
            )
        )
        exit()

    if args.fibonacci:
        # Fibonacci examples
        console.rule("Fibonacci Examples")
        rprint(
            Panel(
                str(interview.fibonacci_recursive(args.fibonacci)),
                title="Fibonacci Recursive",
            )
        )
        rprint(
            Panel(
                str(interview.fibonacci_iterative(args.fibonacci)),
                title="Fibonacci Iterative",
            )
        )
        exit()

    if args.search:
        # Searching examples
        console.rule("Searching Examples")
        array = [1, 3, 5, 7, 9]
        rprint(Panel(str(interview.linear_search(array, 5)), title="Linear Search"))
        rprint(Panel(str(interview.binary_search(array, 5)), title="Binary Search"))
        exit()

    if args.sort:
        # Sorting examples
        console.rule("Sorting Examples")
        unsorted_array = [64, 34, 25, 12, 22, 11, 90]
        rprint(
            Panel(str(interview.bubble_sort(unsorted_array.copy())), title="Bubble Sort")
        )
        rprint(Panel(str(interview.merge_sort(unsorted_array.copy())), title="Merge Sort"))
        exit()

    # Stack example
    console.rule("Stack Example")
    stack = interview.Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    rprint(Panel(str(stack.pop()), title="Stack Pop"))
    rprint(Panel(str(stack.peek()), title="Stack Peek"))
    rprint(Panel(str(stack.size()), title="Stack Size"))

    # Queue example
    console.rule("Queue Example")
    queue = interview.Queue()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    rprint(Panel(str(queue.dequeue()), title="Queue Dequeue"))
    rprint(Panel(str(queue.is_empty()), title="Queue Is Empty"))
    rprint(Panel(str(queue.size()), title="Queue Size"))

    # Linked List example
    console.rule("Linked List Example")
    head = None
    head = interview.insert_linked_list(head, 1)
    head = interview.insert_linked_list(head, 2)
    head = interview.insert_linked_list(head, 3)
    interview.print_linked_list(head)  # Output: 1 -> 2 -> 3 -> None

    # Tree Traversal example
    console.rule("Tree Traversal Example")
    root = interview.TreeNode(1)
    root.left = interview.TreeNode(2)
    root.right = interview.TreeNode(3)
    root.left.left = interview.TreeNode(4)
    root.left.right = interview.TreeNode(5)
    rprint(Panel(str(interview.inorder_traversal(root)), title="Inorder Traversal"))
    rprint(Panel(str(interview.preorder_traversal(root)), title="Preorder Traversal"))
    rprint(Panel(str(interview.postorder_traversal(root)), title="Postorder Traversal"))

    # Graph Algorithms example
    console.rule("Graph Algorithms Example")
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }
    rprint(Panel(str(interview.dfs(graph, "A")), title="DFS"))
    rprint(Panel(str(interview.bfs(graph, "A")), title="BFS"))

    # Starting interactive session with tab completion
    setup_readline(locals())
    banner = "Interactive session started. Type 'exit()' or 'Ctrl-D' to exit."

    if args.interactive:
        code.interact(
            banner=banner, local=locals(), exitmsg="Exiting interactive session."
        )


if __name__ == "__main__":
    main()
