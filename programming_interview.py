from rich import print as rprint
from rich.console import Console

import math
import time


class Interview:
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
        return f"{result}, {elapsed_time:.6f}"

    # Iterative Factorial with Timing
    def factorial_iterative(self, n):
        start_time = time.time()  # Start timing
        result = 1
        for i in range(1, n + 1):
            result *= i
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"{result}, {elapsed_time:.6f}"

    # Divide and Conquer Factorial with Timing
    def factorial_divide_and_conquer(self, n):
        start_time = time.time()  # Start timing
        result = self._factorial_divide_and_conquer(1, n)  # Calculate factorial
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        return f"{result}, {elapsed_time:.6f}"

    # Built-in Factorial with Timing
    def factorial_builtin(self, n):
        start_time = time.time()  # Start timing
        result = math.factorial(n)  # Calculate factorial using built-in
        end_time = time.time()  # End timing

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print complexity and runtime
        return f"{result}, {elapsed_time:.6f}"

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

    # Tree Traversals: Binary Tree
    class TreeNode:
        def __init__(self, value=0, left=None, right=None):
            self.value = value
            self.left = left
            self.right = right

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


console = Console()
interview = Interview()

# Factorial examples
console.rule("Factorial Examples")
rprint("[bold cyan]Recursive Factorial - Time Complexity: O(n)[/bold cyan]")
rprint(interview.factorial_recursive(5))  # Output: 120
rprint("[bold cyan]Iterative Factorial - Time Complexity: O(n)[/bold cyan]")
rprint(interview.factorial_iterative(5))  # Output: 120
rprint("[bold cyan]Built-in Factorial - Time Complexity: O(n)[/bold cyan]")
rprint(interview.factorial_builtin(5))  # Output: 120
rprint(
    "[bold cyan]Divide and Conquer Factorial - Time Complexity: O(n log n)[/bold cyan]"
)
rprint(interview.factorial_divide_and_conquer(5))  # Output: 120

# Fibonacci examples
console.rule("Fibonacci Examples")
rprint(interview.fibonacci_recursive(7))  # Output: 13
rprint(interview.fibonacci_iterative(7))  # Output: 13

# Searching examples
console.rule("Searching Examples")
array = [1, 3, 5, 7, 9]
rprint(interview.linear_search(array, 5))  # Output: 2
rprint(interview.binary_search(array, 5))  # Output: 2

# Sorting examples
console.rule("Sorting Examples")
unsorted_array = [64, 34, 25, 12, 22, 11, 90]
rprint(
    interview.bubble_sort(unsorted_array.copy())
)  # Output: [11, 12, 22, 25, 34, 64, 90]
rprint(
    interview.merge_sort(unsorted_array.copy())
)  # Output: [11, 12, 22, 25, 34, 64, 90]

# Stack example
console.rule("Stack Example")
stack = interview.Stack()
stack.push(1)
stack.push(2)
stack.push(3)
rprint(stack.pop())  # Output: 3
rprint(stack.peek())  # Output: 2
rprint(stack.size())  # Output: 2

# Queue example
console.rule("Queue Example")
queue = interview.Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
rprint(queue.dequeue())  # Output: 1
rprint(queue.is_empty())  # Output: False
rprint(queue.size())  # Output: 2

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
rprint(interview.inorder_traversal(root))  # Output: [4, 2, 5, 1, 3]
rprint(interview.preorder_traversal(root))  # Output: [1, 2, 4, 5, 3]
rprint(interview.postorder_traversal(root))  # Output: [4, 5, 2, 3, 1]

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
rprint(interview.dfs(graph, "A"))  # Output: {'E', 'D', 'A', 'C', 'B', 'F'}
rprint(interview.bfs(graph, "A"))  # Output: {'A', 'B', 'C', 'D', 'E', 'F'}
# code.interact(local=locals())
