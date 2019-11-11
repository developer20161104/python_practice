from function.notebook import Note, Notebook

if __name__ == '__main__':
    n1 = Note('hello first')
    n2 = Note('shut up')

    print(n1.id, n2.id)

    n = Notebook()
    n.new_note("hello")
    n.new_note("again")
    print(n.notes)
