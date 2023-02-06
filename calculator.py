
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class Calculator(App):
    def build(self):
        root_layout = BoxLayout(orientation='vertical')
        self.solution = Label(text=" ", halign="right", size_hint_y=0.75, font_size=50)
        button_symbols = [['7', '8', '9', '+'], 
                        ['4', '5', '6', '-'], 
                        ['1', '2', '3', '*'], 
                        ['.', '0', '/', '=']]
        button_grid = GridLayout(cols=4, size_hint_y=2)
        for row in button_symbols:
            for symbol in row:
                button = Button(text=symbol, pos_hint={'center_x': 0.5, 'center_y': 0.5})
                button_grid.add_widget(button)
        clear_button = Button(text='Clear', size_hint_y=None, height=100, pos_hint={'center_x': 0.5, 'center_y': 0.5})
        
        def print_button_text(instance):
            self.solution.text += instance.text
            
        for button in button_grid.children[1:]:
            button.bind(on_press=print_button_text)
            
        def resize_label_text(label, new_height):
            label.font_size = label.height * 0.75
            
        self.solution.bind(height=resize_label_text)
        
        def evaluate_results(instance):
            try:
                self.solution.text = str(eval(self.solution.text))
            except SyntaxError:
                self.solution.text = "Syntax Error"
        
        button_grid.children[0].bind(on_press=evaluate_results)
        
        def clear_label(instance):
            self.solution.text = " "
            
        clear_button.bind(on_press=clear_label)
        
        root_layout.add_widget(self.solution)
        root_layout.add_widget(button_grid)
        root_layout.add_widget(clear_button)
        return root_layout



if __name__ == '__main__':
    Calculator().run()
