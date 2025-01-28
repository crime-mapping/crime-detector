from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    label = serializers.CharField(max_length=100)
    confidence = serializers.FloatField()
