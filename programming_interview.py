class Interview:
    def __init__(self):
        pass

    # Example of recursion: Factorial
    def factorial_recursive(self, n):
        if n == 0:
            return 1
        return n * self.factorial_recursive(n - 1)

    # Example of iteration: Factorial
    def factorial_iterative(self, n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    # Example of recursion: Fibonacci
    def fibonacci_recursive(self, n):
        if n <= 1:
            return n
        return self.fibonacci_recursive(n - 1) + self.fibonacci_recursive(n - 2)

    # Example of iteration: Fibonacci
    def fibonacci_iterative(self, n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    # Example of searching: Linear Search
    def linear_search(self, arr, target):
        for i, value in enumerate(arr):
            if value == target:
                return i
        return -1

    # Example of searching: Binary Search
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

    # Example of sorting: Bubble Sort
    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    # Example of sorting: Merge Sort
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

    # Example of using a data structure: Stack
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

    # Example of using a data structure: Queue
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

# Example usage
interview = Interview()

# Factorial examples
print(interview.factorial_recursive(5))  # Output: 120
print(interview.factorial_iterative(5))  # Output: 120

# Fibonacci examples
print(interview.fibonacci_recursive(7))  # Output: 13
print(interview.fibonacci_iterative(7))  # Output: 13

# Searching examples
array = [1, 3, 5, 7, 9]
print(interview.linear_search(array, 5))  # Output: 2
print(interview.binary_search(array, 5))  # Output: 2

# Sorting examples
unsorted_array = [64, 34, 25, 12, 22, 11, 90]
print(interview.bubble_sort(unsorted_array.copy()))  # Output: [11, 12, 22, 25, 34, 64, 90]
print(interview.merge_sort(unsorted_array.copy()))   # Output: [11, 12, 22, 25, 34, 64, 90]

# Stack example
stack = interview.Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # Output: 3
print(stack.peek()) # Output: 2
print(stack.size()) # Output: 2

# Queue example
queue = interview.Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(queue.dequeue())  # Output: 1
print(queue.is_empty()) # Output: False
print(queue.size())     # Output: 2
