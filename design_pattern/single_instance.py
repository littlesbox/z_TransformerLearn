# 单例模式
class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.name = ""
        self.age = 0

    def set_name(self, name):
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError
        
    def set_age(self, age):
        if isinstance(age, int):
            self.age = age
        else:
            raise TypeError
        
    def set(self, item, value):
        if item == 'name':
            self.set_name(value)
        elif item == 'age':
            self.set_age(value)
        else:
            raise ValueError
        
    def get(self, item):
        if item == 'name':
            return self.name  
        elif item == 'age':
            return self.age
        else:
            raise ValueError

a = ConfigManager()
a.set('name', 'wang')
print(a.get('name'))
b = ConfigManager()
a is b
print(b.get('name'))
print(ConfigManager._instance)
