# gate-calibrated: plan de implementación

Documento para un agente implementador. Autocontenido. Sin plazos: el orden de las fases es la única secuencia que importa. No saltes la verificación de cada fase.

## 1. Objetivo

Middleware que decide, antes de ejecutar una llamada de escritura de un agente con herramientas, entre:
- **act**: ejecutar sola.
- **ask**: pedir aprobación a un humano.

La decisión usa una probabilidad de acierto calibrada y un umbral con control de riesgo validado en datos separados. Se mide sobre tau-bench (dominio retail) comparando tres políticas con el mismo modelo.

**Resultado a publicar (lo único que importa):** una curva de acciones de escritura erróneas ejecutadas sin aprobación frente a número de aprobaciones pedidas, para las tres políticas, más una tabla de calibración (ECE y Brier antes y después).

**Fuera de alcance (no hacer):** soporte multi-framework, UI, servidor, pasarela MCP, observabilidad, dominio airline/telecom, entrenamiento de modelos.

## 2. Reglas de redacción y honestidad (obligatorias)

- No escribir "primero", "nuevo", "novedoso" ni "state of the art". Posicionamiento permitido: "implementación abierta y reproducible de ideas descritas en ToolChain-CRC (arXiv 2606.18467) y CORA (arXiv 2604.09155), evaluada sobre tau-bench retail".
- Citar tau-bench (MIT, Sierra) y los papers anteriores en el README.
- Todo texto en inglés, sin guiones largos (em/en dash), sin fórmulas típicas de IA, voz concreta. Pasar por el skill /humanizer.
- Sección "Limitations" obligatoria en el README (ver 9).
- `DECISIONS.md` y `AI_REVIEW.md` los firma Youssef. El agente implementador solo mantiene `notes/ai_review_log.md` con el material en bruto (qué generó mal la IA, qué se corrigió y por qué) y borradores de `DECISIONS.md`. No inventar casos.

## 3. Hechos verificados sobre tau-bench (repo `sierra-research/tau-bench`)

- Licencia MIT. Instalación: `pip install -e .` desde el repo clonado. Usa `litellm`.
- `README` indica que existe tau2-bench como sucesor (con dominio telecom). **Decisión:** usar tau-bench original, dominio retail. tau2-bench queda como extensión opcional.
- Bucle del agente en `tau_bench/agents/tool_calling_agent.py`: `completion(...)` -> `message_to_action(message)` -> `env.step(action)`. **El punto de enganche del gate es entre `message_to_action` y `env.step`.**
- Tipos en `tau_bench/types.py`: `Action(name, kwargs)`, `Task(user_id, actions, instruction, outputs)`, `SolveResult(reward, info, messages, total_cost)`, `EnvRunResult(task_id, reward, info, traj, trial)`, `RunConfig`.
- Las acciones doradas de cada tarea están en `Task.actions`. El reward final compara el hash de la base de datos con el estado esperado.
- Existe `historical_trajectories/` (gpt-4o y sonnet-35 en retail). Solo contienen mensajes, **no** tienen confianza declarada ni muestras alternativas. Sirven para probar el etiquetado y el código de tiers, no para entrenar el calibrador.
- Ejecución de referencia: `python run.py --agent-strategy tool-calling --env retail --model <m> --model-provider <p> --user-model <m2> --user-model-provider <p2> --user-strategy llm --max-concurrency N`.

**Verificar al empezar (no asumido):** tamaño de los splits de retail (`train`, `dev`, `test`), nombres exactos de las herramientas en `tau_bench/envs/retail/tools/`, versión de Python soportada, y que la versión instalada de litellm acepte los modelos elegidos.

## 4. Diseño

### 4.1 Estructura del repo

```
gate-calibrated/
  README.md
  DECISIONS.md            (lo escribe Youssef; borrador en notes/)
  AI_REVIEW.md            (lo escribe Youssef)
  LICENSE                 (MIT)
  pyproject.toml
  notes/ai_review_log.md
  gate/
    __init__.py
    types.py              dataclasses: StepRecord, GateDecision
    tiers.py              tier de riesgo por herramienta
    signals.py            señales de cada paso
    labeling.py           etiqueta de acierto por acción
    calibrator.py         temperature scaling y regresión logística
    threshold.py          umbral con control de riesgo
    policy.py             políticas: confirm_all, fixed_threshold, calibrated
    middleware.py         hook sobre el bucle de tau-bench + nodo LangGraph
    oracle.py             humano simulado
  eval/
    collect.py            ejecuta el agente base y guarda trazas con señales
    run_policies.py       ejecuta las tres políticas en vivo
    metrics.py            pass^k, errores, aprobaciones, ECE, Brier
    report.py             curva y tablas en results/
  tests/
  results/                (solo artefactos finales reproducibles)
```

### 4.2 Tiers de herramientas (`tiers.py`)

Verificar nombres contra `tau_bench/envs/retail/tools/` antes de codificar. Esperado:
- **read** (act siempre, sin gate): find_user_id_by_email, find_user_id_by_name_zip, get_order_details, get_product_details, get_user_details, list_all_product_types, calculate.
- **write** (gate): modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address.
- **irreversible** (ask siempre, también en la política calibrada): cancel_pending_order, exchange_delivered_order_items, return_delivered_order_items.
- **special**: `respond` (mensaje al usuario) y transfer_to_human_agents: pasan sin gate.

Si la verificación contradice esta lista, manda el código del repo; anótalo en `notes/`.

### 4.3 Etiquetado (`labeling.py`)

Para una acción de tier write o irreversible propuesta en una tarea:
- `correct = True` si existe en `Task.actions` una acción con el mismo `name` y los mismos `kwargs` (comparación canónica: claves ordenadas, listas de items ordenadas cuando el orden no importe, strings normalizados).
- `correct = False` en otro caso.
- Las acciones read no se etiquetan.

Test obligatorio: la lista dorada de una tarea, reproducida paso a paso, etiqueta todo como correcto.

### 4.4 Señales (`signals.py`)

Se calculan solo para acciones write (las irreversibles siempre piden, pero se registran las señales igual para analizar).
1. **verbalized**: una llamada extra al mismo modelo con los mensajes y la acción propuesta: "probabilidad entre 0 y 1 de que esta acción sea correcta según la política". Salida estructurada, un solo número. Temperatura 0.
2. **agreement**: k muestras (por defecto k=5, temperatura 0.7) de la siguiente acción con los mismos mensajes. Valor = fracción cuya acción canónica (name + kwargs) coincide con la propuesta.
3. **tier**: one-hot.
4. **step_index**: número de pasos previos en la conversación.

Registrar también tokens y coste de cada llamada de señal. Claude no devuelve logprobs, así que no depender de ellos.

### 4.5 Calibrador (`calibrator.py`)

Dos variantes, ambas con la misma interfaz `fit(X, y)` / `predict_proba(X)`:
- **A: temperature scaling** sobre el logit de `verbalized`. Un parámetro T.
- **B: regresión logística** sobre [logit(verbalized), agreement, tier, step_index].
Métricas en `metrics.py`: ECE con 10 bins de igual masa, Brier, diagrama de fiabilidad. Reportar crudo (verbalized sin calibrar), A y B, siempre en el conjunto de test.

### 4.6 Umbral con control de riesgo (`threshold.py`)

- Pérdida por trayectoria t para un umbral λ: `L_t(λ) = 1` si en la trayectoria algún paso write habría sido ejecutado sin aprobación (`p_hat >= λ`) y era erróneo; `0` en otro caso. Es no creciente en λ.
- Calibración con N trayectorias: elegir el menor λ tal que `(N/(N+1)) * mean(L_t(λ)) + 1/(N+1) <= alpha` (control de riesgo conforme).
- La unidad es la trayectoria, no el paso, porque los pasos de una misma trayectoria no son intercambiables.
- Valores de alpha a barrer: 0.02, 0.05, 0.10, 0.20.
- Documentar la limitación: la calibración usa trayectorias sin gate; en el despliegue el gate cambia el camino (el humano rechaza y el agente replanifica). La garantía es una aproximación; la evaluación en vivo (4.8) mide el efecto real.

### 4.7 Políticas (`policy.py`)

Interfaz: `decide(step_record) -> "act" | "ask"`.
- **confirm_all**: ask en toda acción write e irreversible.
- **fixed_threshold**: ask si `verbalized < 0.7` (umbral típico a mano de los tutoriales), irreversible siempre ask.
- **calibrated**: ask si `p_hat < lambda_alpha` o tier irreversible.

### 4.8 Oráculo y evaluación en vivo (`oracle.py`, `run_policies.py`)

Humano simulado: cuando la política dice ask, el oráculo aprueba si la acción es correcta según 4.3 y la rechaza si no. Al rechazar, se inserta un mensaje de herramienta con "action rejected by reviewer" y el agente continúa. Cada ask cuenta como una aprobación pedida.

Para cada política y cada alpha: ejecutar las tareas del split de test con `num_trials >= 4` y registrar:
- `executed_wrong_writes`: acciones write erróneas ejecutadas sin pasar por ask.
- `approvals_requested`.
- `pass_hat_k` (pass^k, k=1..4) con el estimador no sesgado `E[C(c,k)/C(n,k)]`.
- coste en tokens y en dinero.

### 4.9 Integración LangGraph (`middleware.py`)

Dos formas, ambas delegan en `policy.decide`:
1. `TauGateAgent`: subclase de `ToolCallingAgent` que llama al gate entre `message_to_action` y `env.step`. Es la que se usa en la evaluación.
2. `gate_node(state)`: nodo LangGraph que, si la política dice ask, llama a `interrupt()` con la acción, las señales y el p_hat; se reanuda con `Command(resume=...)`. Incluye un ejemplo mínimo ejecutable en `examples/` y un test que comprueba pausa y reanudación.

La evaluación no depende de LangGraph; la integración demuestra que el gate funciona como middleware.

## 5. Particiones y reproducibilidad

- Dividir **por task_id**, nunca por paso. Tres bloques disjuntos: `fit` (entrenar calibrador), `cal` (elegir umbral), `test` (evaluar en vivo). Si el split de test de tau-bench es el único con tareas suficientes, usar `train`/`dev` para fit y cal y `test` para la evaluación.
- Semillas fijas y guardadas en un archivo de configuración. Todas las ejecuciones escriben JSONL con la config completa.
- Cachear llamadas de litellm en disco para poder repetir el informe sin volver a pagar.
- Modelos configurables por variables de entorno. Registrar versión exacta del modelo en cada resultado.

## 6. Fases y verificación (en orden estricto)

**Fase 0. Entorno.**
- Crear repo, `pyproject.toml`, instalar tau-bench como dependencia editable en un entorno virtual.
- Verificar los puntos de "Verificar al empezar" de la sección 3 y anotarlos en `notes/env_check.md`.
- Verificación: ejecutar una sola tarea de retail con el agente base y obtener un `reward`.

**Fase 1. Tiers y etiquetado.**
- Implementar `tiers.py` y `labeling.py` con tests (incluido el de lista dorada).
- Verificación: sobre `historical_trajectories/*-retail.json`, etiquetar todas las acciones write y contar. Reportar la tasa de acciones write erróneas de cada modelo histórico. Este número da la tasa base del problema.

**Fase 2. Recolección con señales.**
- Implementar `signals.py` y `eval/collect.py`.
- Ejecutar el agente base sobre las tareas de `fit` y `cal` con `num_trials >= 2`, guardando por cada paso write: acción, señales, etiqueta, coste.
- Verificación: ninguna fila sin etiqueta; distribución de `verbalized` y `agreement` revisada con histogramas. Si `verbalized` es casi constante (por ejemplo siempre >0.9), anotarlo; es un hallazgo, no un fallo.

**Fase 3. Calibrador.**
- Implementar variantes A y B con tests unitarios (datos sintéticos con calibración conocida).
- Verificación: ECE y Brier en `cal`. Reportar si calibrar mejora frente a crudo. Si no mejora, reportarlo igual.

**Fase 4. Umbral.**
- Implementar `threshold.py` con test: con p_hat sintéticos y etiquetas generadas a partir de p_hat, la tasa empírica de pérdida en una muestra nueva es <= alpha en la gran mayoría de repeticiones (simulación de Monte Carlo en el test).
- Verificación: tabla alpha -> lambda -> aprobaciones esperadas en `cal`.

**Fase 5. Políticas, oráculo y middleware.**
- Implementar `policy.py`, `oracle.py`, `TauGateAgent` y `gate_node`.
- Verificación: test de integración con un modelo simulado (stub determinista) que proponga una acción errónea y una correcta; el oráculo rechaza la primera y aprueba la segunda. Test de `interrupt()`/resume.

**Fase 6. Evaluación en vivo.**
- Ejecutar las tres políticas y los alpha definidos en `test`, `num_trials >= 4`.
- Verificación: confirm_all debe tener `executed_wrong_writes` cercano a 0 en acciones write y el máximo de aprobaciones; si no, hay un bug en el gate.

**Fase 7. Informe.**
- `eval/report.py`: curva (eje x aprobaciones por trayectoria, eje y errores write ejecutados por trayectoria), tabla de calibración, tabla pass^k, intervalos de confianza por bootstrap sobre tareas.
- `README.md` con el resultado principal en tres frases, cómo reproducir, tabla de costes, Limitations.

## 7. Criterios de aceptación

- `pip install -e .` y `pytest` pasan desde cero.
- Un comando reproduce el informe a partir de las trazas cacheadas.
- La curva incluye las tres políticas con intervalos.
- ECE y Brier de crudo, A y B en el mismo conjunto de test.
- Ningún texto contiene afirmaciones de primacía.
- Los resultados negativos (calibrar no mejora, la curva no mejora a confirm_all, etc.) se reportan tal cual.

## 8. Control de coste

- Primero todo con un modelo barato y pocas tareas para verificar el pipeline; solo después el modelo final.
- Las señales (verbalized + k muestras) se calculan solo en pasos write, no en todos.
- Registrar el coste acumulado en cada ejecución y detener si supera un tope definido en la config (`MAX_USD`).
- Esta fase no tiene presupuesto fijado: **pedir a Youssef el tope antes de la Fase 2.**

## 9. Limitations (texto mínimo para el README)

- El humano es un oráculo perfecto con las acciones doradas de tau-bench; una persona real se equivoca y se cansa.
- tau-bench retail es un dominio y una política concretos; no se evalúa generalización.
- El usuario es simulado por un modelo.
- La garantía de riesgo asume intercambiabilidad entre trayectorias de calibración y de test y se calcula sin gate; en vivo se mide aparte.
- El etiquetado compara con acciones doradas; puede haber acciones distintas pero válidas que se cuentan como erróneas.
- Resultados con un modelo y un conjunto de semillas.

## 10. Preguntas abiertas para Youssef (antes de empezar)

1. Tope de gasto en API y modelo preferido para agente, para señales y para el usuario simulado.
2. Nombre final del repo y si va bajo `Gjusev`.
3. ¿Hace falta incluir el modo `abstain` (el agente pregunta al usuario en lugar de pedir aprobación)? Por defecto: no, queda para después de la Fase 7.
