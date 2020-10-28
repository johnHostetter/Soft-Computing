import numpy as np

class PseudoTrapezoidMembershipFunction:
    def __init__(self, a, b, c, d, H, I, D, sigma=None):
        """ The argument a must be less than or equal to the argument b.
        The argument b must be less than or equal to the argument c.
        The argument c must be less than or equal to the argument d.
        The argument H is strictly greater than zero, but less than or equal to one,
        and determines the membership degree when within the interval [b, c].
        The argument I is a nondecreasing function of x on a domain [a, b)
        that returns values within the interval [0, 1]. The argument D is a 
        nonincreasing function in (c, d] that returns values in the interval [0, 1].
        The argument sigma (optional) is used for the Gaussian special case.
    
        If H = 1, then the fuzzy set is normal.
    
        The trapezoid membership function is a special case when: 
            I(x) = (x - a) / (b - a)
        and
            D(x) = (x - d) / (c - d) .
        
        The triangular membership function is a special case when:
            I(x) = (x - a) / (b - a) ,
            D(x) = (x - d) / (c - d) ,
        and
            b = c .
            
        The Gaussian membership function is a special case when:
            a = -inf ,
            b = c = mean(x) ,
            d = inf ,
        and
            I(x) = D(x) = exp(- pow(((x - mean(x)) / sigma), 2)) .
            
        Citation:
            A Course in Fuzzy Systems and Control by Li-Xin Wang
            Chapter 10.1 Preliminary Concepts pg. 129
        """
        if a <= b and b <= c and c <= d and (0 < H and H <= 1):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.H = H
            self.I = I
            self.D = D
            self.sigma = sigma
        else:
            raise ValueError('The definition of Pseudo-Trapezoid Membership Function has been violated.')
    def trapezoid_nondecreasing(self, x):
        return (x - self.a) / (self.b - self.a)
    def trapezoid_nonincreasing(self, x):
        return (x - self.d) / (self.c - self.d)
    def gaussian(self, x):
        if not self.b == self.c:
            raise ValueError('The values for b and c must be equal.')
        if self.sigma == None:
            raise ValueError('The value of sigma has not been defined for the Gaussian membership function.')
        return np.exp(- pow(((x - self.b) / self.sigma), 2))
    def degree(self, x):
        if self.a <= x and x < self.b:
            return self.I(self, x)
        elif self.b <= x and x <= self.c:
            return self.H
        elif self.c < x and x <= self.d:
            return self.D(self, x)
        elif x < self.a or self.d < x:
            return 0.0
        else:
            raise ValueError('Invalid argument for x when calling function degree().')

"""
The following code was written by Seyed Saeid Masoumzadeh (GitHub user ID: seyedsaeidmasoumzadeh),
and was published for public use on GitHub under the MIT License at the following link:
    https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning 
"""

class Trapeziums(object):
    def __init__(self, left, left_top, right_top, right):
        self.left = left
        self.right = right
        self.left_top = left_top
        self.right_top = right_top

    def membership_value(self, input_value):
        if (input_value >= self.left_top) and (input_value <= self.right_top):
            membership_value = 1.0
        elif (input_value <= self.left) or (input_value >= self.right_top):
            membership_value = 0.0
        elif input_value < self.left_top:
            membership_value = (input_value - self.left) / (self.left_top - self.left)
        elif input_value > self.right_top:
            membership_value = (input_value - self.right) / (self.right_top - self.right)
        else:
            membership_value = 0.0
        return membership_value

class Triangles(object):
    def __init__(self, left, top, right):
        self.left = left
        self.right = right
        self.top = top

    def membership_value(self, input_value):
        if input_value == self.top:
            membership_value = 1.0
        elif input_value <= self.left or input_value >= self.right:
            membership_value = 0.0
        elif input_value < self.top:
            membership_value = (input_value - self.left) / (self.top - self.left)
        elif input_value > self.top:
            membership_value = (input_value - self.right) / (self.top - self.right)
        return membership_value