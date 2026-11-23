"""Binary min-heap with operation counters.

Written by hand (not heapq) so that heap operations can be counted for the
experiments. Shared by Dijkstra, A* and (Member 2) the race-event queue.
Items are compared with Python's normal tuple ordering, so a key such as
(triggerTime, priority, insertionSequence, payload...) works directly.
Complexity: push O(log n), pop O(log n), peek O(1).
"""


class MinHeap:
    def __init__(self):
        self._a = []
        self.pushes = 0
        self.pops = 0
        self.comparisons = 0

    def __len__(self):
        return len(self._a)

    def __bool__(self):
        return bool(self._a)

    @property
    def ops(self):
        return self.pushes + self.pops

    def reset_counters(self):
        self.pushes = self.pops = self.comparisons = 0

    def peek(self):
        if not self._a:
            raise IndexError("peek from empty heap")
        return self._a[0]

    def push(self, item):
        self._a.append(item)
        self._sift_up(len(self._a) - 1)
        self.pushes += 1

    def pop(self):
        a = self._a
        if not a:
            raise IndexError("pop from empty heap")
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            self._sift_down(0)
        self.pops += 1
        return top

    def _sift_up(self, i):
        a = self._a
        item = a[i]
        while i > 0:
            p = (i - 1) >> 1
            self.comparisons += 1
            if item < a[p]:
                a[i] = a[p]
                i = p
            else:
                break
        a[i] = item

    def _sift_down(self, i):
        a = self._a
        n = len(a)
        item = a[i]
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            r = c + 1
            if r < n:
                self.comparisons += 1
                if a[r] < a[c]:
                    c = r
            self.comparisons += 1
            if a[c] < item:
                a[i] = a[c]
                i = c
            else:
                break
        a[i] = item
