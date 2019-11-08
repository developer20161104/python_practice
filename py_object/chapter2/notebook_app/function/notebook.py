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
