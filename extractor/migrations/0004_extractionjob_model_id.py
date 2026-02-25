from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("extractor", "0003_add_viz_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="extractionjob",
            name="model_id",
            field=models.CharField(
                blank=True,
                default="",
                max_length=100,
                help_text="LLM model used for extraction (e.g. 'gemini-2.0-flash', 'fast', 'glm')",
            ),
        ),
    ]
