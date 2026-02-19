"""
Models for Clone Hero Content Manager
Note: These models map to existing database tables created by the API service.
No migrations needed - we're using the existing schema.
"""
from django.db import models


class Song(models.Model):
    """
    Maps to the existing 'songs' table in PostgreSQL.
    This is a read-only model for displaying song data.
    """
    id = models.AutoField(primary_key=True)
    title = models.TextField()
    artist = models.TextField()
    album = models.TextField(blank=True, null=True)
    file_path = models.TextField(unique=True)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'songs'  # Use existing table
        managed = False  # Don't let Django manage this table
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.artist}"

    def get_metadata_display(self):
        """Return non-empty metadata fields for display"""
        return {k: v for k, v in self.metadata.items() if v}
