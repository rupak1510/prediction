document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const area = document.getElementById('area').value;
    const date = document.getElementById('date').value;
    const time = document.getElementById('time').value;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ area, date, time }),
        });
        
        const result = await response.json();
        if (result.error) {
            document.getElementById('output').textContent = result.error;
            return;
        }
        
        const predictedPower = result.prediction.toFixed(2);
        document.getElementById('output').textContent = `Predicted Power for ${area} on ${date} at ${time}: ${predictedPower} mw`;
    } catch (error) {
        document.getElementById('output').textContent = 'Error: Could not get prediction';
        console.error('Error:', error);
    }
});
