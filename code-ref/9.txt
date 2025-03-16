class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        i=0
        j=0
        n=len(nums1)
        m=len(nums2)
        m1=0
        m2=0
        ans=0

        if n == 0:

            mid = m // 2
            if m % 2 == 0:
                return (nums2[mid] + nums2[mid - 1]) / 2.0
            else:
                return float(nums2[mid])
       
        if m == 0:
            mid = n // 2
            if n % 2 == 0:
                return (nums1[mid] + nums1[mid - 1]) / 2.0
            else:
                return float(nums1[mid])
        
        for count in range((n + m) // 2 + 1):
            m2 = m1
            
            if i < n and (j >= m or nums1[i] <= nums2[j]):
                m1 = nums1[i]
                i += 1
            else:
                m1 = nums2[j]
                j += 1

        if (n + m) % 2 == 1:
            return float(m1)
        else:
            return (m1 + m2) / 2.0