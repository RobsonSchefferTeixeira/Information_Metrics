class Formatter:
    @staticmethod
    def as_currency(value):
        return f"${value:,.2f}"

    @staticmethod
    def as_percent(value):
        return f"{value:.2%}"

# Usage
# formatter = Formatter()
# print(formatter.as_currency(1000))  # Output: $1,000.00
# print(formatter.as_percent(0.25))  # Output: 25.00%