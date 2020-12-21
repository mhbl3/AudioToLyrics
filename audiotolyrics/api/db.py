import peewee as pw
from datetime import datetime
import config
from playhouse.shortcuts import model_to_dict


db = pw.PostgresqlDatabase(
    config.POSTGRES_DB,
    user=config.POSTGRES_USER, password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST, port=config.POSTGRES_PORT
)

class BaseModel(pw.Model):
    class Meta:
        database = db

# Table Description
class GenLyrics(BaseModel):

    filename = pw.TextField()
    time_song = pw.IntegerField()
    start_word = pw.TextField()
    gen_lyrics = pw.TextField()
    # ip_address = pw.TextField()
    created_date = pw.DateTimeField(default=datetime.now)

    def serialize(self):
        review_dict = model_to_dict(self)
        review_dict["created_date"] = (
            review_dict["created_date"].strftime('%Y-%m-%d %H:%M:%S')
        )

        return review_dict


# Connection and table creation
db.connect()
db.create_tables([GenLyrics])