def sub():
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    print(f"The Subtraction of {b} from {a} is {a-b}")

def add():
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    print(f"The Addition of {a} and {b} is {a+b}")

def mul():
    a = int(input("Enter the first number: "))
    b = int(input("Enter the second number: "))
    print(f"The Multiplication of {a} and {b} is {a*b}")

while True:
    print("Welcome to calculator...")
    print("(1) Addition  (2) Subtraction  (3) Multiplication  (4) Exit")
    x = int(input("Enter Your Choice: "))
    if x==1:
      add()
    elif x==2:
      sub()
    elif x==3:
      mul()
    elif x==4:
      print("exited for calculator")
      break
    else:print("Enter valid number!")







