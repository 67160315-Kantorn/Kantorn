class BubbleSorter:
    def __init__(self, nums):
        self.nums = nums

    def display(self):
        print(f"Current data: {self.nums}")

    def sort(self):
        n = len(self.nums)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if self.nums[j] > self.nums[j + 1]:
                    self.nums[j], self.nums[j + 1] = self.nums[j + 1], self.nums[j]
            print(f"After round {i + 1}: {self.nums}")


if __name__ == "__main__":
    nums = [64, 34, 25, 12, 22, 11, 90]
    sorter = BubbleSorter(nums)

    print("Before sorting:")
    sorter.display()

    sorter.sort()

    print("After sorting:")
    sorter.display()