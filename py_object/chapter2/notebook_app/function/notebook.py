import datetime

# 为所有笔记存储下一个可用id
last_id = 0


class Note:
    """represent a note in the notebook,
    match against a string in searches and store tags for each one"""

    def __init__(self, memo, tags=''):
        """initialize a note with memo and optional space-separated tags.
        automatically set the note's creation date and a unique id"""
        self.memo = memo
        self.tags = tags
        self.creation_date = datetime.date.today()

        global last_id
        last_id += 1
        self.id = last_id

    def match(self, filters):
        """determine if this note matches the filter text
        return true if it matches, false otherwise
        search is case sensitive and matches both text and tags"""
        return filters in self.memo or filters in self.tags


class Notebook:
    """represent a collection of notes that can be tagged, modified, and searched
    """

    def __init__(self):
        """Initial a notebook with an empty list
        """

        self.notes = []

    def new_note(self, memo, tags=''):
        """create a new note and add it to the list"""
        self.notes.append(Note(memo, tags))

    def _find_note(self, note_id):
        """locate the note with given id"""
        for note in self.notes:
            if str(note.id) == str(note_id):
                return note
        return None

    def modify_memo(self, note_id, memo):
        """find the note with the given id and change its memo to the given value
        """
        """there is a bug for None value"""
        note = self._find_note(note_id)
        if note:
            note.memo = memo
            return True
        return False

    def modify_tags(self, note_id, tags):
        """find the note with the given id and change its tags to the given value
        """
        for note in self.notes:
            if note.id == note_id:
                note.tags = tags
                break

    def search(self, filters):
        """find all notes that match the given filter string
        """
        return [note for note in self.notes if note.match(filters)]