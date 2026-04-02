# Políticas de Uso — Modelo de Riesgo Metabólico Bolivia
**Versión:** 1.0 | **Fecha:** 2026 | **Proyecto:** MLOps EDSA Bolivia

---

## 1. Descripción del Modelo

Este sistema predice el riesgo metabólico de mujeres bolivianas en edad reproductiva
(clasificación del perímetro de cintura en tres categorías según estándar OMS),
entrenado con datos del dataset EDSA 2023 del INE Bolivia.

**Clases de salida:**
- Clase 0 — Sin riesgo (cintura estimada ≤ 80 cm)
- Clase 1 — Riesgo leve (cintura estimada 81-88 cm)
- Clase 2 — Riesgo alto (cintura estimada > 88 cm)

---

## 2. Usos Permitidos

- Tamizaje investigativo en salud pública a nivel poblacional
- Análisis de prevalencia de riesgo metabólico por departamento, zona geográfica y área
- Priorización de poblaciones para programas de intervención preventiva
- Investigación académica sobre factores de riesgo metabólico en Bolivia
- Evaluación del impacto de programas de salud a nivel poblacional

---

## 3. Usos Prohibidos

- **Diagnóstico clínico individual:** el modelo NO reemplaza evaluación médica
- **Decisiones administrativas sobre pacientes individuales** (acceso a servicios, seguros)
- Uso con poblaciones fuera del perfil de entrenamiento (hombres, niños, otros países)
- Uso como única base para intervenciones médicas sin validación clínica adicional
- Comercialización del modelo o sus predicciones sin autorización explícita

---

## 4. Limitaciones Conocidas

- Entrenado exclusivamente con datos bolivianos (EDSA 2023) — no generalizable
  a otras poblaciones sin revalidación
- El target (riesgo por cintura) usa umbrales OMS diseñados para poblaciones europeas;
  su aplicabilidad en alta altitud requiere validación específica
- La variable Anemia imputa 0 para mujeres sin medición de hemoglobina (~73%)
- Dataset de corte transversal: no permite inferir causalidad ni evolución temporal
- Precisión en Clase 1 (Riesgo Leve) es menor que en las otras clases por desbalance

---

## 5. Versionado de Modelos

| Versión  | Año encuesta | Estado      | Fecha retiro |
|----------|-------------|-------------|--------------|
| v1.0     | EDSA 2023   | Production  | Al publicar EDSA 2025 |
| v2.0     | EDSA 2024   | Staging     | —            |

**Política de versiones (semver simplificado):**
- `v1.0` → primera versión productiva
- `v2.0` → nueva encuesta EDSA (cambio mayor)
- `v1.1` → corrección de bug sin nueva encuesta (cambio menor)

---

## 6. Política de Retiro Automático

Un modelo es retirado de producción automáticamente si:
- F1-Macro cae más de 5% respecto a la versión anterior en el conjunto de evaluación
- Se detecta drift severo (IMC media cambia >10% del baseline)
- Se identifica un error de implementación que afecte las predicciones

---

## 7. Transparencia

- Todas las versiones del modelo y sus métricas son accesibles en el MLflow local
- Cada respuesta de la API incluye la versión del modelo y la advertencia de uso
- El código fuente completo está disponible en el repositorio del proyecto

---

## 8. Contacto

**Autora:** Alexandra Cristal Salazar Gisberth  
**Institución:** UMSA — Carrera de Informática  
**Materia:** Machine Learning 2026
