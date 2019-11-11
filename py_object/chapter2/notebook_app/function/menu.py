import sys
from function.notebook import Notebook, Note


class Menu:
    """display a menu and respond to choices when run
    """

    def __init__(self):
        self.notebook = Notebook()
        self.choices = {
            "1": self.show_notes,
            "2": self.search_notes,
            "3": self.add_note,
            "4": self.modify_note,
            "5": self.quit
        }

    def display_menu(self):
        print("""
1. show all notes
2. search notes
3. add note
4.modify note
5. quit
        """)

    def run(self):
        """display the menu and respond to choices
        """
        while True:
            self.display_menu()
            choice = input("Enter an option: ")
            action = self.choices.get(choice)
            if action:
                action()
            else:
                print("{0} is not a valid choice ".format(choice))

    def show_notes(self, notes=None):
        if not notes:
            notes = self.notebook.notes
        for note in notes:
            print("{0}: {1}\n{2}".format(
                note.id, note.tags, note.memo
            ))

    def search_notes(self):
        filters = input("search for: ")
        notes = self.notebook.search(filters)
        self.show_notes(notes)

    def add_note(self):
        memo = input("enter a memo: ")
        self.notebook.new_note(memo)
        print("your note has been added.")

    def modify_note(self):
        id = input("enter a note id: ")
        memo = input("enter a memo: ")
        tags = input("enter tags: ")
        if memo:
            self.notebook.modify_memo(id, memo)
        else:
            self.notebook.modify_tags(id, tags)

    def quit(self):
        print("thank you for using your notebook today.")
        sys.exit(0)


if __name__ == '__main__':
    Menu().run()
