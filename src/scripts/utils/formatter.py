class Formatter:
    _MAX_LENGTH = 50
    _current_line = ""

    def __init__(self):
        self.line_length = 0
        while not self.line_length:
            self.line_length = self.get_line_length()
            self.confirm_line_length()

    def get_line_length(self):
        user_input = input(
            f"Enter the number of '_'s and the last number in the first "
            f"line, must be in (1-{10*Formatter._MAX_LENGTH-1})\n"
            f"{self._scale}\n:"
        )

        try:
            line_lenght = int(user_input)
            assert self.line_length in range(10 * Formatter._MAX_LENGTH)
            return line_lenght
        except ValueError:
            print(f"Could not parse '{user_input}' as number. please try " "again.")
        except AssertionError:
            print(
                f"'{self.line_length}' is not in range({10 * Formatter._MAX_LENGTH})"
            )
        return 0

    def confirm_line_length(self):
        user_input = input(
            f"Line length is set to {self.line_length}, the following "
            f"sequence of '='s should not have any line breaks\n"
            f"{'='*self.line_length}\nEverything looking good? (y/N):"
        )
        if user_input.lower() != "y":
            self.line_length = 0

    def print(self, message: str):
        paragraphs = [
            self.format_paragraph(paragraph) for paragraph in message.split("\n")
        ]
        print("\n\n".join(paragraphs))

    def format_paragraph(self, paragraph: str):
        if len(paragraph) <= self.line_length:
            return paragraph
        words = paragraph.split(" ")
        output = ""
        current_line = ""
        for word in words:
            word = word.replace("\t", " " * 4)  # replace TAB with four spaces
            # check if adding next the next makes the line too long
            if len(current_line) + len(word) > self.line_length:
                output += current_line + "\n"
                current_line = ""
            current_line += " " + word

        # add remaining words to output
        if current_line:
            output += current_line

        return output

    @property
    def _scale(self):
        one_to_nine = "123456789_"
        return one_to_nine * Formatter._MAX_LENGTH
