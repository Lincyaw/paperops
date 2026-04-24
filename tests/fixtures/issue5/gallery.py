from paperops.slides.dsl import Deck, Grid, Heading, Title


DECK = Deck(theme="minimal")
DECK.slide(
    Title("Gallery"),
    Heading("Overview"),
    Grid(style={"cols": "1fr 1fr", "gap": "md"})[
        Heading("Cell A"),
        Heading("Cell B"),
    ],
)
