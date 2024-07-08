

from sympy import symbols, exp, sinh, cosh, diff, Matrix, Abs, sign
from sympy import simplify, DiracDelta
# Define symbols
x = symbols('x', real=True)

a = symbols('a', real=True, positive=True)
l = symbols('l', real=True, positive=True)

# Define the function f(x, y)
phi = exp(-abs(x)/l) + 1/(exp(2*a/l)+1) * 2*sinh(abs(x)/l) 

gradient_x = -exp(-Abs(x)/l)*sign(x)/l + 2*cosh(Abs(x)/l)*sign(x)/(l*(exp(2*a/l) + 1))

gradient2_x = exp(-Abs(x)/l)*sign(x)**2/l**2 + 2*sinh(Abs(x)/l)*sign(x)**2/(l**2*(exp(2*a/l) + 1))


check = 1/l * phi - l*(gradient2_x)
print(simplify(check))

#print(simplify(check)==0)

# # Multiply by l
# result = l * (f_xx + f_yy)

# # Simplify the result
# simplified_result = result.simplify()

# # Print the simplified result
# print(simplified_result)

