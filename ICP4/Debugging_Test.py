# Python program to find the factorial of a number provided by the user.
# change the value for a different result
num = 7
# To take input from the user
num1 = int(input("Enter a first number:"))
num2 = int(input("Enter a second number:"))
num3 = int(input("Enter a third number:"))

def factorial(num):
    factorial = 1
    # check if the number is negative, positive or zero
    if num < 0:
        print("Sorry, factorial does not exist for negative numbers")
    elif num == 0:
        print("The factorial of 0 is 1")
    else:
        for i in range(1,num + 1):
            factorial = factorial*i

    result = factorial
    return result

print("First time calling ")
result = factorial(num1)
print("The factorial of {} is {}".format(num1,result))
print("second time calling ")
result = factorial(num2)
print("The factorial of {} is {}".format(num2,result))
print("third time calling ")
result = factorial(num3)
print("The factorial of {} is {}".format(num3,result))