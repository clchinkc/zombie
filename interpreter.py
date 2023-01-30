
class Interpreter:
    def __init__(self):
        self.stack = []
        #存储变量映射关系的字典变量
        self.environment = {}

    def STORE_NAME(self, name):
        val = self.stack.pop()
        self.environment[name] = val

    def LOAD_NAME(self, name):
        val = self.environment[name]
        self.stack.append(val)

    def LOAD_VALUE(self, number):
        self.stack.append(number)

    def PRINT_ANSWER(self):
        answer = self.stack.pop()
        print(answer)

    def ADD_TWO_VALUES(self):
        first_num = self.stack.pop()
        second_num = self.stack.pop()
        total = first_num + second_num
        self.stack.append(total)

    def parse_argument(self, instruction, argument, what_to_execute):
        #解析命令参数
        #使用常量列表的方法
        numbers = ["LOAD_VALUE"]
        #使用变量名列表的方法
        names = ["LOAD_NAME", "STORE_NAME"]

        if instruction in numbers:
            argument = what_to_execute["numbers"][argument]
        elif instruction in names:
            argument = what_to_execute["names"][argument]

        return argument

    def execute(self, what_to_execute):
        instructions = what_to_execute["instructions"]
        for each_step in instructions:
            instruction, argument = each_step
            argument = self.parse_argument(instruction, argument, what_to_execute)
            bytecode_method = getattr(self, instruction)
            if argument is None:
                bytecode_method()
            else:
                bytecode_method(argument)

# python code
def s():
    a = 1
    b = 2
    print(a + b)

# Instruction code / code object
what_to_execute = {
    "instructions": [("LOAD_VALUE", 0),
                    ("STORE_NAME", 0),
                    ("LOAD_VALUE", 1),
                    ("STORE_NAME", 1),
                    ("LOAD_NAME", 0),
                    ("LOAD_NAME", 1),
                    ("ADD_TWO_VALUES", None),
                    ("PRINT_ANSWER", None)],
    "numbers": [1, 2],
    "names":   ["a", "b"] }

if __name__ == '__main__':
    interpreter = Interpreter()
    interpreter.execute(what_to_execute)

"""
https://github.com/nedbat/byterun
https://github.com/python/cpython/blob/main/Python/ceval.c
https://github.com/aosabook/500lines
https://www.51cto.com/article/517141.html
https://zhuanlan.zhihu.com/p/45101508
https://zhuanlan.zhihu.com/p/481884570
"""