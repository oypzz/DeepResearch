from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def isPalindrome( head: Optional[ListNode]) -> bool:
    dummy_node = ListNode(next=head)
    val_list = []
    while dummy_node.next:
        val_list.append(dummy_node.next)
        dummy_node = dummy_node.next
    left, right = 0, len(val_list) - 1
    while left < right:
        if val_list[left] == val_list[right]:
            left += 1
            right -= 1
        else:
            return False
    return True
head = [1,2,2,1]
print(isPalindrome(head))
