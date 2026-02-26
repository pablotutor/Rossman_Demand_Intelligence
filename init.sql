-- Creamos la tabla para guardar el histórico y las predicciones
CREATE TABLE IF NOT EXISTS daily_forecasts (
    id SERIAL PRIMARY KEY,
    target_date DATE NOT NULL,
    item_id VARCHAR(50) NOT NULL,
    actual_value FLOAT,        -- El valor real (si lo sabemos)
    predicted_value FLOAT,     -- Lo que dijo nuestro modelo
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);