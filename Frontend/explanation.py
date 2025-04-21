import streamlit as st

st.title("📝 Explicación: Algoritmo Genético Adaptativo para Distancia entre Paneles Solares")

st.markdown("""
**Objetivo base**: Optimizar la distancia mínima entre paneles solares fotovoltaicos del tipo A-135P (1476×659×35 mm) instalados a una latitud de 41°, inclinados 45°, para evitar sombras en invierno y reducir distancia en verano.

Este problema se plantea en el PDF del Taller AGA, donde se define:

- Cálculo de la elevación solar crítica:
  - Invierno: $\\alpha_{inv}=(90^\circ-\\varphi)-23.45^\circ$
  - Verano:  $\\alpha_{ver}=(90^\circ-\\varphi)+23.45^\circ$
- Fórmula de distancia mínima:
  $D_M = B\cos\\beta + \\frac{B\sin\\beta}{\\tan\\alpha}$
  
  Donde:
  - **B**: longitud del panel (1476 mm)
  - **β**: ángulo de inclinación (45°)
  - **α**: elevación solar crítica (invierno/verano)
""", unsafe_allow_html=True)

st.header("1. Representación y Fitness")
st.markdown("""
- **Cromosoma**: un valor real `D` (distancia en mm).
- **Fitness**: mide la cercanía al valor óptimo teórico:

```python
def fitness(D, D_opt):
    return abs(D - D_opt)
```
""", unsafe_allow_html=True)

st.header("2. Selección por Torneo")
st.markdown("Se eligen aleatoriamente un subconjunto de la población y se selecciona el individuo con menor fitness (porque se busca la distancia minima):")
st.code("""def ranking_selection(pob, D_opt, pct):
    import random
    sub = random.sample(pob, int(len(pob)*pct))
    return min(sub, key=lambda x: abs(x - D_opt))
""", language='python')

st.header("3. Cruce y Mutación")
st.subheader("3.1 Cruce Aritmético")
st.markdown("""
             Simple de implementar y genera hijos en el rango $[P_1, P_2]$ , el problema es que puede reducir la diversidad si siempre mezcla de la misma forma.

  Ecuación: 
  $H_1 = α * P_1 + (1 - α)P_2$, $H_2 = (1 - α)P_1 + α * P_2$

  donde α es un valor fijo o aleatorio entre 0 y 1

  referencia: https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
  """)
st.code("""def arithmetic_crossover(p1, p2, alpha=None):
    import random
    if alpha is None:
        alpha = random.random()
    return (alpha*p1 + (1-alpha)*p2,
            (1-alpha)*p1 + alpha*p2)
""", language='python')

st.subheader("3.2 BLX-α")
st.markdown("""
            
  Para cada cruce, extiende el intervalo $[P_{min}, P_{max}]$ en un factor α

  $ I = P_{min} - P_{max} $ y tambien se calcula el $margen = α * I$ 

  luego se muestrea cada hijo uniformemente en $[P_{min} - margen, P_{max} - margen]$

  el valor de α es entre (0,1)
  tiene un problema cuando α es muy grande por que los hijos se alejan mucho agregando ruido.
  
  referencia: http://www.tomaszgwiazda.com/blendX.htm
    """)

st.code("""def blx_alpha(p1, p2, alpha=0.3):
    import random
    lo, hi = min(p1,p2), max(p1,p2)
    d = hi - lo
    return (random.uniform(lo - alpha*d, hi + alpha*d),
            random.uniform(lo - alpha*d, hi + alpha*d))
""", language='python')

st.subheader("3.3 SBX (Simulated Binary Crossover)")
st.markdown("""
              En este método se imitan cruces binarios usando un índice de distribución ƞ.
  se calcula un factor β según una distribucióbn coontrolada por ƞ.

  este método es muy usado en GAs reales gracias a que balancea exploración/explotación según ƞ que normalmente tiene vales entre 2 y 5.

  Ecuación:
  $h_1 = 0.5[(1 + β) * p_1 + (1 - β) * p_2 ]$ y $h_2 = 0.5[(1 - β) * p_1 + (1 + β) * p_2 ]$

  reference: https://www.researchgate.net/publication/220742263_Self-adaptive_simulated_binary_crossover_for_real-parameter_optimization
    """)

st.code("""def sbx(p1, p2, eta=2):
    import random
    u = random.random()
    if u <= 0.5:
        beta = (2*u)**(1/(eta+1))
    else:
        beta = (1/(2*(1-u)))**(1/(eta+1))
    return (0.5*((1+beta)*p1 + (1-beta)*p2),
            0.5*((1-beta)*p1 + (1+beta)*p2))
""", language='python')

st.subheader("3.4 Mutación Gaussiana")
st.markdown("""
              Suma ruido gaussiano truncado al individuo original para no salirse de los límites.
  
  Ecuación:
  $D' = D + N(0,δ)$ ⇒ Truncado a $[D_{min},D_m{max}]$

  este método es útil porque permite un control fino de la magnitud de la perturbación con δ aunque si este valor es muy grande puede comportarse como mutación aleatoria.
    """)
st.code("""def gaussian_mutation(D, D_min, D_max, sigma=500):
    import random
    Dp = D + random.gauss(0, sigma)
    return min(max(Dp, D_min), D_max)
""", language='python')

st.subheader("3.5 Mutación Polinómica")
st.markdown("""
            Se Utiliza una distribución polinómica controlada por un índice $ƞ_m$
  para generar mutaciones preferentemente pequeñas.

  para esto se hacen los siguientes calculos:
  1. Generar un valor u entre (0,1)
  2. Calcular el δ para la mutación mediante:
  """)
st.latex(r"""
  \begin{equation}\delta =
  \begin{cases}
  (2u)^{\frac{1}{\eta_m + 1}} - 1, & \text{si } u < 0.5 \\
  1 - [2(1 - u)]^{\frac{1}{\eta_m + 1}}, & \text{si } u \geq 0.5
  \end{cases}
  \end{equation}
  """)
  
st.markdown("""
  3. Calcular la mutación
  """)

st.latex(r"""
           \begin{equation}D' = D + δ x 
  \begin{cases}
    D - D_{min}, & δ < 0 \\
    D_{max} - D, & δ ≥ 0 
  \end{cases}
  \end{equation}
    """)
  
st.code("""def polynomial_mutation(D, D_min, D_max, eta_m=20):
    import random
    u = random.random()
    if u < 0.5:
        delta = (2*u)**(1/(1+eta_m)) - 1
    else:
        delta = 1 - (2*(1-u))**(1/(1+eta_m))
    if delta < 0:
        Dp = D + delta*(D - D_min)
    else:
        Dp = D + delta*(D_max - D)
    return min(max(Dp, D_min), D_max)
""", language='python')

st.header("4. Adaptación de Parámetros")
st.markdown("""
Existen diferentes métodos de adaptación pero se decide implementar 2 (basado en diversidad de población y basado en tasa de mejora del fitness) para que el usuario escoga cual de las dos opciones desea utilizar para hacer adaptativas las probabilidades de mutación y cruce.

1. **Basado en diversidad**:
    
    La idea es medir la diversidad 'D' como la desviación estándar media de cada individuo en la población, por lo que las probabilidades de mutación ($p_m$) y cruce ($p_c$) se ajustan según:
""")
st.latex(r"""
     \begin{equation}
   p_m(D) = p_{m,min} + (p_{m,max} - p_{m,min}) e^{-k_mD} 
    \end{equation}
    """)
st.latex(r"""
    \begin{equation}
   p_c(D) = p_{c,max} + (p_{c,max} - p_{c,min}) e^{-k_cD}
  \end{equation}
""")      
st.markdown("""
      Entonces cuando la población es muy homogenea ('D' es pequeño), $P_m$ tiende hacia su valor maximo y $P_c$ tiende a menos valor para menor cruce, por el contrario si la población es diversa el valor de la probabilidad de cruce aumenta y la de mutación disminuye.
            
   ```python
   import statistics, math
   D = statistics.pstdev(poblacion)
   pm = pm_min + (pm_max-pm_min)*math.exp(-k_m*D)
   pc = pc_max - (pc_max-pc_min)*math.exp(-k_c*D)
   ```

2. **Basado en tasa de mejora**:

  En este método la idea es calcular la mejora relativa del fitness promedio entre generaciones:
  
    """)
st.latex(r"""
           \begin{equation}
    ΔF = \frac{f_{t-1} - f_t}{f_{t-1}}
  \end{equation}
""")
st.markdown("""
  luego se ajusta con:
    """)
st.latex(r"""
  \begin{equation}
    p_m(ΔF) = p_{m,min} + (p_{m,max} - p_{m,min}) (1 - ΔF) 
    \end{equation}
""")
st.latex(r"""
\begin{equation}
    p_c(ΔF) = p_{c,min} + (p_{c,max} - p_{c,min}) ΔF
  \end{equation}
""")
st.markdown("""
   ```python
   fit_prev = avg_fitness(poblacion, D_opt)
   fit_curr = avg_fitness(nueva_pob, D_opt)
   deltaF = (fit_prev - fit_curr)/fit_prev
   pm = pm_min + (pm_max-pm_min)*(1 - deltaF)
   pc = pc_min + (pc_max-pc_min)*deltaF
   ```
""", unsafe_allow_html=True)

