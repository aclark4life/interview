from ninja import Schema

class URLSchema(Schema):
    url: str

class URLStatsSchema(Schema):
    original_url: str
    access_count: int
