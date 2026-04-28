# # a = [1,2,3]
# a = range(100)
# print(a*0)
from collections import deque
from typing import List


# def fourSumCount( nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
#     from collections import defaultdict
#     map = defaultdict(int)
#     counter = 0
#     for i in range(len(nums1)):
#         for j in range(len(nums2)):
#             key = nums1[i] + nums2[j]
#             map[key] += 1
#     for k in range(len(nums3)):
#         for l in range(len(nums4)):
#             cd = nums3[k] + nums4[l]
#             if 0 - cd in map:
#                 counter += map[0 - cd]
#     return counter
#
# nums1 = [1,2]
# nums2 = [-2,-1]
# nums3 = [-1,2]
# nums4 = [0,2]
# print(fourSumCount(nums1,nums2,nums3,nums4))

def search( nums: List[int], target: int) -> int:
    n = len(nums)
    if nums[0] > target:
        return -1
    i = int(n / 2)
    while nums[i] != target:
        if nums[i] > target and i >= 1:
            i -= 1
        elif nums[i] < target and i <= n -1 :
            i += 1
        else:
            return -1
    return i

def searchMatrix( nums: List[int], target: int) :
    list = sorted(nums)
    left, right = 0, len(nums)
    while left < right:
        middle = left + (right - left) // 2
        if list[middle] > target:
            right = middle
        elif list[middle] < target:
            left = middle - 1
        else:
            return middle
    return -1

def findMin(nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        min_num = nums[left]
        while left <= right:
            middle = (left + right) // 2
            if nums[left] < nums[middle]:
                min_num = min(nums[left], min_num)
                left = middle + 1
            else:
                min_num = min(nums[middle], min_num)
                left = middle +1
        return min_num


def isValid( s: str) -> bool:
    if s == "":
        return True
    stack = []
    # 后遇见的括号要先闭合 因此使用栈
    map = {
        '[': ']',
        '(': ')',
        '{': '}'
    }
    for ch in s:
        if ch in map:
            if ch in map.keys():
                stack.append(map[ch])

        elif not stack or stack[-1] != ch:
            return False
        else:
            stack.pop()
    return not stack

def maxSlidingWindow( nums: List[int], k: int) -> List[int]:
    list = []
    q = deque()
    # 1.入
    for i, x in enumerate(nums):
        # 当前元素大于队尾元素就弹出
        while q and nums[q[-1]] <= x:
            q.pop()
        q.append(i)
        # 2.出
        # 只维护长度为k的窗口，超长则去除队首
        if i - q[0] >= k:
            q.popleft()
        # 3.记录答案
        if i >= k - 1:
            list.append(nums[q[0]])
    return list


def topKFrequent(nums: List[int], k: int) -> List[int]:
    map = {}
    final_list = []
    for i in range(len(nums)):
        if str(nums[i]) not in map:
            map[str(nums[i])] = 1
        else:
            map[str(nums[i])] += 1
    fre_list = sorted(list(map.values()))
    for j in range(len(fre_list) - 1, -1, -1):
        for key in map.keys():
            if map[key] == fre_list[j]:
                final_list.append(int(key))
    return final_list[:k]


class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.min_stack:
            if self.min_stack[-1] >= val:
                # self.min_stack.pop()
                self.min_stack.pop()
                self.min_stack.append(val)
            else:

                self.min_stack.append(self.min_stack[-1])

        else:
            # self.min_stack.pop()
            self.min_stack.append(val)

    def pop(self) -> None:

        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            self.stack.pop()
        else:
            return None

    def top(self) -> int:
        if self.stack:
            return self.stack[-1]

    def getMin(self) -> int:
        if self.stack:
            return self.min_stack[-1]

def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
        # 类似的下一个更大更小问题都可以考虑使用单调栈
        len1 = len(nums1)
        ans = [-1] * len1
        stack = []
        mapping = {}
        for i in range(len(nums2)):
            if not stack:
                stack.append(nums2[i])
            while stack and nums2[i] > stack[-1]:
                mapping[stack[-1]] = nums2[i]
                stack.pop()
            stack.append(nums2[i])

            mapping[stack[-1]] = -1
        for j in range(len(nums1)):
            ans[j] = mapping[nums1[j]]
        return ans


class MedianFinder:

    def __init__(self):
        self.data = []

    def addNum(self, num: int) -> None:
        self.data.append(num)

    def findMedian(self) -> float:
        total_len = len(self.data)

        def quick_select(data, k):

            import random

            small, equal, big = [], [], []
            pivot = random.choice(self.data)
            for data in self.data:
                if data > pivot:
                    big.append(data)
                elif data < pivot:
                    small.append(data)
                else:
                    equal.append(data)

            if total_len % 2 == 0:
                k = total_len / 2
            else:
                k = total_len / 2 + 1

            if k <= len(big):
                # 第 k 大元素在 big 中，递归划分
                return quick_select(big, k)
            if k > len(big) + len(equal):
                # 第 k 大元素在 small 中，递归划分
                return quick_select(small, k - len(big) - len(equal))
            return pivot

        if total_len % 2 == 0:
            k1 = total_len / 2
            k2 = total_len / 2 + 1
            return (quick_select(self.data, k1) + quick_select(self.data, k2)) / 2
        else:
            k = total_len / 2 + 1
            return quick_select(self.data, k)



# median = MedianFinder()
# median.data = [1,2]
# print(median.findMedian())
# nums1 = [4,1,2]
# nums2 = [1,3,4,2]
# print(len(matrix[0]))
# print(nextGreaterElement(nums1,nums2))

import sys


    # print(int(a[0]) + int(a[1]))

def scientificCount(N: int):
    str_num = str(N)
    c = len(str_num) - 1

    float_ab = N / 10 ** c
    ab = str(round(float_ab,ndigits=1))
    # a = int(ab.split('.')[0])
    # b = int(ab.split('.')[1])

    return ab +  "*10^" + str(c)

def partitionLabels( s: str) -> List[int]:
        char_map = {}

        ans = []
        for i in range(len(s)):
            char_map[s[i]] = i  # 记录每个字母出现的最远下标
        min_idx = char_map[s[0]]
        start_idx = 0
        for j in range(len(s)):
            current_idx = char_map[s[j]]
            min_idx = max(current_idx, min_idx)
            if j >= min_idx:
                if ans:
                    ans.append(min_idx + 1 - start_idx)
                else:
                    ans.append(min_idx + 1)
                start_idx = min_idx + 1
        return ans
# s = "ababcbacadefegdehijhklij"
# print(partitionLabels(s))
# for line in sys.stdin:
#     a = line.split()
#     print(scientificCount(a))

# num_line = int(input())
# for _ in range(num_line):
#     line = input()
#     n = int(line.split(' ')[0])
#     sum = 0
#     for i  in range(n):
#         sum += int(line.split(' ')[i+1])
#     print(sum)
#     print('')

def rob(nums: List[int]) -> int:
    dp = [0] * len(nums)
    # dp[0] = 0
    dp[0] = nums[0]
    dp[1] = max(dp[0],nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[len(nums) - 1]
print(rob([1,1]))

import heapq
min_heapq = []
heapq.heappush(min_heapq,10)
print(min_heapq[0])
heapq.heappop(min_heapq)