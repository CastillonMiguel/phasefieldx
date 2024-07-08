

from sympy import symbols, exp, sinh, cosh, diff, Matrix, Abs, sign
from sympy import simplify, DiracDelta, sech, cosh
# Define symbols
x = symbols('x', real=True)
y = symbols('y', real=True)
d = 3#symbols('d', real=True, positive=True)
l = 1#symbols('l', real=True, positive=True)

# Define the function f(x, y)
#phi = exp(-abs(y)/l) + 1/exp(2*(d-abs(x))/l) * 2*sinh(abs(y)/l) 
phi = sech((d-x)/l) * cosh((-d+x+y)/l)
# gradient_x = diff(phi, x)
# gradient_y = diff(phi, y)

#gradient = Matrix([gradient_x, gradient_y])

gradient_x =  4*exp(-(2*d - 2*Abs(x))/l)*sinh(Abs(y)/l)*sign(x)/l
gradient_y = -exp(-Abs(y)/l)*sign(y)/l + 2*exp(-(2*d - 2*Abs(x))/l)*cosh(Abs(y)/l)*sign(y)/l

gradient2_x = 8*exp((-2*d + 2*Abs(x))/l)*sinh(Abs(y)/l)*DiracDelta(x)/l + 8*exp((-2*d + 2*Abs(x))/l)*sinh(Abs(y)/l)*sign(x)**2/l**2
gradient2_y = 4*exp((-2*d + 2*Abs(x))/l)*cosh(Abs(y)/l)*sign(x)*sign(y)/l**2

df_dx2 = diff(phi, x, 2)
df_dy2 = diff(phi, y, 2)

# Calculate the Laplacian
laplacian_f = df_dx2 + df_dy2

check = 1/l * phi + l*(laplacian_f)
print(check==0)
print(check)
print(simplify(check))

# # Multiply by l
# result = l * (f_xx + f_yy)

# # Simplify the result
# simplified_result = result.simplify()

# # Print the simplified result
# print(simplified_result)

