from PIL import Image, ImageDraw, ImageFont
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import argparse
import locale
import math
import time

import code  # noqa
import readline  # noqa
import rlcompleter  # noqa


locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


class BigONotationPlotter:
    def __init__(self):
        self.console = Console()
        self.functions = {
            "O(1)": self.constant,
            "O(log n)": self.logarithmic,
            "O(n)": self.linear,
            "O(n log n)": self.linearithmic,
            "O(n^2)": self.quadratic,
            "O(n^3)": self.cubic,
            "O(2^n)": self.exponential,
        }
        self.colors = {
            "O(1)": "red",
            "O(log n)": "green",
            "O(n)": "blue",
            "O(n log n)": "orange",
            "O(n^2)": "purple",
            "O(n^3)": "brown",
            "O(2^n)": "black",
        }

    def constant(self, n):
        return 1

    def logarithmic(self, n):
        return math.log(n)

    def linear(self, n):
        return n

    def linearithmic(self, n):
        return n * math.log(n)

    def quadratic(self, n):
        return n**2

    def cubic(self, n):
        return n**3

    def exponential(self, n):
        return 2**n

    def generate_data_points_for_rich(self, n_values):
        table = Table(title="Big O Notations")

        # Add columns
        table.add_column("n", justify="right", style="cyan", no_wrap=True)
        for label in self.functions.keys():
            table.add_column(label, justify="right", style="magenta")

        # Add rows
        for n in n_values:
            row = [str(n)]
            for func in self.functions.values():
                try:
                    row.append(f"{func(n):.2f}")
                except OverflowError:
                    row.append("∞")
            table.add_row(*row)

        return table

    def generate_data_points_for_pillow(self, n_values):
        data_points = {}
        for label, func in self.functions.items():
            data_points[label] = [func(n) for n in n_values]
        return data_points

    def plot_rich(self, n_values):
        table = self.generate_data_points_for_rich(n_values)
        self.console.print(table)

    def plot_pillow(
        self, n_values, width=800, height=600, output_file="big_o_notations.png"
    ):
        data_points = self.generate_data_points_for_pillow(n_values)
        max_n = max(n_values)
        max_y = max(max(points) for points in data_points.values())

        # Create an image with white background
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Draw axes
        margin = 50
        draw.line(
            (margin, height - margin, width - margin, height - margin), fill="black"
        )
        draw.line((margin, height - margin, margin, margin), fill="black")

        # Draw labels
        font = ImageFont.load_default()
        draw.text((width // 2, height - margin + 10), "n", fill="black", font=font)
        draw.text((10, height // 2), "f(n)", fill="black", font=font)

        # Plot each function
        for label, points in data_points.items():
            color = self.colors[label]
            for i in range(len(n_values) - 1):
                x1 = margin + (n_values[i] / max_n) * (width - 2 * margin)
                y1 = height - margin - (points[i] / max_y) * (height - 2 * margin)
                x2 = margin + (n_values[i + 1] / max_n) * (width - 2 * margin)
                y2 = height - margin - (points[i + 1] / max_y) * (height - 2 * margin)
                draw.line((x1, y1, x2, y2), fill=color, width=2)

        # Add legend
        legend_x = width - margin - 150
        legend_y = margin
        for label, color in self.colors.items():
            draw.text((legend_x, legend_y), label, fill=color, font=font)
            legend_y += 15

        # Save the image
        image.save(output_file)


class DataStructure:
    """
    Data Structures for Programming Interview Questions
    """

    # Data Structure: Binary Tree
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


class Factorial:
    """
    Factorial Algorithms for Programming Interview Questions
    """

    # Iterative Factorial with Timing
    def factorial_iterative(self, n):
        start_time = time.time()  # Start timing
        result = 1
        for i in range(1, n + 1):
            result *= i
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}\n  Elapsed time: {elapsed_time:.6f}"

    # Protected method for recursive factorial calculation
    def _factorial_recursive(self, n):
        if n == 0:
            return 1
        return n * self._factorial_recursive(n - 1)

    # Recursive Factorial with Timing
    def factorial_recursive(self, n):
        start_time = time.time()  # Start timing
        result = self._factorial_recursive(n)  # Calculate factorial
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}\n  Elapsed time: {elapsed_time:.6f}"

    # Built-in Factorial with Timing
    def factorial_builtin(self, n):
        start_time = time.time()  # Start timing
        result = math.factorial(n)  # Calculate factorial using built-in
        end_time = time.time()  # End timing

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print complexity and runtime
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}\n  Elapsed time: {elapsed_time:.6f}"

    # Protected method for recursive factorial calculation
    def _factorial_recursive_divide_and_conquer(self, low, high):
        if low > high:
            return 1
        if low == high:
            return low
        mid = (low + high) // 2
        return self._factorial_recursive_divide_and_conquer(
            low, mid
        ) * self._factorial_recursive_divide_and_conquer(mid + 1, high)

    # Divide and Conquer Factorial with Timing
    def factorial_recursive_divide_and_conquer(self, n):
        start_time = time.time()  # Start timing
        result = self._factorial_recursive_divide_and_conquer(1, n)
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"  Factorial: {locale.format_string("%.2f", result, grouping=True)}\n  Elapsed time: {elapsed_time:.6f}"


class Fibonacci:
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


class Search:
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


class Sort:
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


class List:
    """
    Linked List Algorithms for Programming Interview Questions
    """

    # Insert into Linked List
    def insert_linked_list(self, head, value):
        new_node = self.ListNode(value)
        if not head:
            return new_node
        current = head
        while current.next:
            current = current.next
        current.next = new_node
        return head

    # Print Linked List
    def print_linked_list(self, head):
        current = head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")


class Traversal:
    """
    Tree Traversal Algorithms for Programming Interview Questions
    """

    # Tree Traversal: Inorder, Preorder, Postorder
    def inorder_traversal(self, root):
        return (
            self.inorder_traversal(root.left)
            + [root.value]
            + self.inorder_traversal(root.right)
            if root
            else []
        )

    # Tree Traversal: Preorder
    def preorder_traversal(self, root):
        return (
            [root.value]
            + self.preorder_traversal(root.left)
            + self.preorder_traversal(root.right)
            if root
            else []
        )

    # Tree Traversal: Postorder
    def postorder_traversal(self, root):
        return (
            self.postorder_traversal(root.left)
            + self.postorder_traversal(root.right)
            + [root.value]
            if root
            else []
        )


class Graph:
    """
    Graph Algorithms for Programming Interview Questions
    """

    # Graph Algorithms: Depth-First Search

    def dfs(self, graph, start):
        print("Graph: ", graph)
        print("Start: ", start)

        # Initialize an empty set to keep track of visited nodes
        visited = set()

        # Initialize a stack with the start node
        stack = [start]

        # Loop until the stack is empty
        while stack:
            # Pop a node from the stack
            vertex = stack.pop()
            print(f"Vertex: {vertex}")

            # If the node has not been visited
            if vertex not in visited:
                # Mark the node as visited
                print(f"Visited: {visited}")
                visited.add(vertex)

                # Add all unvisited neighbors to the stack
                print(f"Unvisited: {set(graph[vertex]) - visited}")
                stack.extend(set(graph[vertex]) - visited)

        # Return the set of visited nodes
        return visited

    # Graph Algorithms: Breadth-First Search
    def bfs(self, graph, start):
        print("Graph: ", graph)
        print("Start: ", start)

        # Initialize a set to keep track of visited nodes
        visited = set()
        # Initialize the queue with the start node
        queue = [start]

        # Continue the loop as long as there are nodes in the queue
        while queue:
            # Dequeue the first node from the queue
            vertex = queue.pop(0)
            print(f"Vertex: {vertex}")

            # If the node has not been visited
            if vertex not in visited:
                # Mark the node as visited
                print(f"Visited: {visited}")
                visited.add(vertex)
                # Add all unvisited neighbors of the current node to the queue
                print(f"Unvisited: {set(graph[vertex]) - visited}")
                queue.extend(set(graph[vertex]) - visited)

        # Return the set of visited nodes after BFS traversal is complete
        return visited


class Interview(
    DataStructure, Factorial, Fibonacci, Search, Sort, List, Traversal, Graph
):
    """
    Programming Interview Questions
    """


def main():
    console = Console()
    interview = Interview()

    parser = argparse.ArgumentParser(description="Programming Interview Questions")

    parser.add_argument(
        "-f", "--factorial", type=int, help="Factorial algorithm examples"
    )
    parser.add_argument("--fibonacci", type=int, help="Fibonacci algorithm examples")
    parser.add_argument(
        "--search", action="store_true", help="Search algorithm examples"
    )
    parser.add_argument("--sort", action="store_true", help="Search algorithm examples")
    parser.add_argument("--stack", action="store_true", help="Stack algorithm examples")
    parser.add_argument("--queue", action="store_true", help="Queue algorithm examples")
    parser.add_argument(
        "--list", action="store_true", help="Linked List algorithm examples"
    )
    parser.add_argument(
        "--tree", action="store_true", help="Tree traversal algorithm examples"
    )
    parser.add_argument("--graph", action="store_true", help="Graph algorithm examples")
    parser.add_argument("-o", "--big-o", type=int, help="Big O Notation plotter")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )

    args = parser.parse_args()

    if args.big_o:
        plotter = BigONotationPlotter()
        n_values = list(range(1, args.big_o + 1))
        plotter.plot_pillow(n_values)
        print(plotter.plot_rich(n_values))

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
                + str(interview.factorial_recursive_divide_and_conquer(args.factorial)),
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
            Panel(
                str(interview.bubble_sort(unsorted_array.copy())), title="Bubble Sort"
            )
        )
        rprint(
            Panel(str(interview.merge_sort(unsorted_array.copy())), title="Merge Sort")
        )
        exit()

    if args.stack:
        # Stack example
        console.rule("Stack Example")
        stack = interview.Stack()
        stack.push(1)
        stack.push(2)
        stack.push(3)
        rprint(Panel(str(stack.pop()), title="Stack Pop"))
        rprint(Panel(str(stack.peek()), title="Stack Peek"))
        rprint(Panel(str(stack.size()), title="Stack Size"))

    if args.queue:
        # Queue example
        console.rule("Queue Example")
        queue = interview.Queue()
        queue.enqueue(1)
        queue.enqueue(2)
        queue.enqueue(3)
        rprint(Panel(str(queue.dequeue()), title="Queue Dequeue"))
        rprint(Panel(str(queue.is_empty()), title="Queue Is Empty"))
        rprint(Panel(str(queue.size()), title="Queue Size"))

    if args.list:
        # Linked List example
        console.rule("Linked List Example")
        head = None
        head = interview.insert_linked_list(head, 1)
        head = interview.insert_linked_list(head, 2)
        head = interview.insert_linked_list(head, 3)
        interview.print_linked_list(head)  # Output: 1 -> 2 -> 3 -> None

    if args.tree:
        # Tree Traversal example
        console.rule("Tree Traversal Example")
        root = interview.TreeNode(1)
        root.left = interview.TreeNode(2)
        root.right = interview.TreeNode(3)
        root.left.left = interview.TreeNode(4)
        root.left.right = interview.TreeNode(5)
        rprint(Panel(str(interview.inorder_traversal(root)), title="Inorder Traversal"))
        rprint(
            Panel(str(interview.preorder_traversal(root)), title="Preorder Traversal")
        )
        rprint(
            Panel(str(interview.postorder_traversal(root)), title="Postorder Traversal")
        )

    if args.graph:
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

    if args.interactive:
        # Starting interactive session with tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer(locals()).complete)
        banner = "Interactive programming interview session started. Type 'exit()' or 'Ctrl-D' to exit."
        code.interact(
            banner=banner,
            local=locals(),
            exitmsg="Great interview!",
        )


if __name__ == "__main__":
    main()
