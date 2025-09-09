import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import math
import os
from datetime import datetime

# ==================== CLASE PARA PRUEBAS ESTADÍSTICAS ====================
class StatisticalTests:
    @staticmethod
    def norm_ppf(p):
        """Aproximación del percentil de la distribución normal"""
        if p < 0.5:
            return -StatisticalTests.norm_ppf(1-p)
        else:
            t = math.sqrt(-2 * math.log(1-p))
            c0 = 2.515517
            c1 = 0.802853
            c2 = 0.010328
            d1 = 1.432788
            d2 = 0.189269
            d3 = 0.001308
            return t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3)
    
    @staticmethod
    def chi2_ppf(p, df):
        """Aproximación del percentil de chi-cuadrado"""
        if df == 1:
            return (-2 * math.log(1 - p)) ** 0.5
        else:
            return df * (1 - 2/(9*df) + StatisticalTests.norm_ppf(p) * math.sqrt(2/(9*df))) ** 3

    @staticmethod
    def media_test(numeros, confianza=0.95):
        n = len(numeros)
        media = np.mean(numeros)
        z_alpha = StatisticalTests.norm_ppf(1 - (1 - confianza) / 2)
        li = 0.5 - z_alpha * (1 / np.sqrt(12 * n))
        ls = 0.5 + z_alpha * (1 / np.sqrt(12 * n))
        pasa_prueba = li <= media <= ls
        return media, li, ls, z_alpha, pasa_prueba

    @staticmethod
    def varianza_test(numeros, confianza=0.95):
        n = len(numeros)
        varianza = np.var(numeros)
        alpha = 1 - confianza
        chi2_inf = StatisticalTests.chi2_ppf(alpha/2, n-1)
        chi2_sup = StatisticalTests.chi2_ppf(1-alpha/2, n-1)
        li = chi2_inf / (12 * (n - 1))
        ls = chi2_sup / (12 * (n - 1))
        pasa_prueba = li <= varianza <= ls
        return varianza, li, ls, chi2_inf, chi2_sup, pasa_prueba

    @staticmethod
    def uniformidad_test(numeros, intervalos=10, confianza=0.95):
        n = len(numeros)
        frec_obs, bins = np.histogram(numeros, bins=intervalos, range=(0, 1))
        frec_esp = n / intervalos
        chi2_calculado = np.sum((frec_obs - frec_esp)**2 / frec_esp)
        grados_libertad = intervalos - 1
        chi2_critico = StatisticalTests.chi2_ppf(confianza, grados_libertad)
        pasa_prueba = chi2_calculado <= chi2_critico
        return frec_obs, frec_esp, chi2_calculado, chi2_critico, grados_libertad, bins, pasa_prueba

# ==================== CLASE PARA GENERACIÓN DE NÚMEROS ====================
class NumberGeneration:
    @staticmethod
    def cuadrados_medios(semilla, n):
        numeros = []
        historial = []
        x = semilla
        
        for i in range(n):
            cuadrado = x * x
            str_cuadrado = str(cuadrado)
            
            # Asegurar que tenga longitud par
            if len(str_cuadrado) % 2 != 0:
                str_cuadrado = '0' + str_cuadrado
            
            # Extraer dígitos del medio (según el ejemplo)
            medio = len(str_cuadrado) // 2
            inicio = medio - 2
            fin = medio + 2
            
            if inicio < 0:
                inicio = 0
            if fin > len(str_cuadrado):
                fin = len(str_cuadrado)
            
            nuevo_num = int(str_cuadrado[inicio:fin])
            numero_aleatorio = nuevo_num / 10000.0
            
            historial.append({
                'iteracion': i+1,
                'yi': x,
                'yi_cuadrado': cuadrado,
                'yi_estrella': nuevo_num,
                'ri': numero_aleatorio
            })
            
            numeros.append(numero_aleatorio)
            x = nuevo_num
            if x == 0:
                break
        
        return numeros, historial

    @staticmethod
    def productos_medios(semilla1, semilla2, n):
        numeros = []
        historial = []
        x0 = semilla1
        x1 = semilla2
        
        for i in range(n):
            producto = x0 * x1
            str_producto = str(producto)
            
            # Asegurar longitud mínima de 4 dígitos
            if len(str_producto) < 4:
                str_producto = str_producto.zfill(4)
            
            # Extraer 4 dígitos del medio
            longitud = len(str_producto)
            inicio = (longitud - 4) // 2
            fin = inicio + 4
            
            if inicio < 0:
                inicio = 0
            if fin > len(str_producto):
                fin = len(str_producto)
            
            medio_str = str_producto[inicio:fin]
            nuevo_num = int(medio_str) if medio_str else 0
            numero_aleatorio = nuevo_num / 10000.0
            
            historial.append({
                'iteracion': i+1,
                'yi0': x0,
                'yi1': x1,
                'producto': producto,
                'yi_estrella': nuevo_num,
                'ri': numero_aleatorio
            })
            
            numeros.append(numero_aleatorio)
            x0 = x1
            x1 = nuevo_num
            if x1 == 0:
                break
        
        return numeros, historial

    @staticmethod
    def multiplicador_constante(semilla, constante, n):
        numeros = []
        historial = []
        x = semilla
        
        for i in range(n):
            producto = constante * x
            str_producto = str(producto)
            
            # Asegurar que tenga longitud par
            if len(str_producto) % 2 != 0:
                str_producto = '0' + str_producto
            
            # Extraer dígitos del medio (según el ejemplo)
            medio = len(str_producto) // 2
            inicio = medio - 2
            fin = medio + 2
            
            if inicio < 0:
                inicio = 0
            if fin > len(str_producto):
                fin = len(str_producto)
            
            nuevo_num = int(str_producto[inicio:fin])
            numero_aleatorio = nuevo_num / 10000.0
            
            historial.append({
                'iteracion': i+1,
                'yi': x,
                'constante': constante,
                'producto': producto,
                'yi_estrella': nuevo_num,
                'ri': numero_aleatorio
            })
            
            numeros.append(numero_aleatorio)
            x = nuevo_num
            if x == 0:
                break
        
        return numeros, historial

# ==================== CLASE PRINCIPAL DE LA APLICACIÓN ====================
class RandomNumberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Generación de Números Pseudoaleatorios")
        self.root.geometry("1000x800")
        self.root.configure(bg="#0a041a")
        
        # Estilo futurista
        self.colors = {
            "bg_dark": "#0a041a",
            "bg_medium": "#1a0c38",
            "bg_light": "#2a1460",
            "accent": "#8a2be2",  # Violeta
            "neon": "#bf00ff",    # Magenta neón
            "text": "#ffffff",
            "highlight": "#00ffff"  # Cian neón
        }
        
        # Configurar estilo
        self.setup_styles()
        
        # Variables para almacenar parámetros
        self.n = tk.IntVar(value=15)
        self.confianza_medias = tk.DoubleVar(value=0.95)
        self.confianza_varianza = tk.DoubleVar(value=0.95)
        self.confianza_uniformidad = tk.DoubleVar(value=0.95)
        self.intervalos_chi = tk.IntVar(value=10)
        
        # Variables específicas para métodos
        self.semilla1_cuadrados = tk.IntVar(value=5115)
        self.semilla1_medios = tk.IntVar(value=1234)
        self.semilla2_medios = tk.IntVar(value=5678)
        self.semilla_multiplicador = tk.IntVar(value=1234)
        self.constante_multiplicador = tk.IntVar(value=5678)
        
        # Lista para almacenar números generados
        self.numeros_generados = []
        self.historial_generacion = []
        self.metodo_actual = ""
        
        # Mostrar menú principal al inicio
        self.mostrar_menu_principal()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores
        style.configure('TFrame', background=self.colors["bg_dark"])
        style.configure('TLabel', background=self.colors["bg_dark"], foreground=self.colors["text"], font=('Roboto', 10))
        style.configure('Title.TLabel', background=self.colors["bg_dark"], foreground=self.colors["neon"], font=('Roboto', 16, 'bold'))
        style.configure('Neon.TLabel', background=self.colors["bg_dark"], foreground=self.colors["neon"], font=('Roboto', 10, 'bold'))
        
        # Configurar botones
        style.configure('Neon.TButton', 
                       background=self.colors["bg_light"],
                       foreground=self.colors["text"],
                       bordercolor=self.colors["neon"],
                       focuscolor=self.colors["bg_medium"],
                       font=('Roboto', 10, 'bold'),
                       padding=(10, 5))
        style.map('Neon.TButton',
                 background=[('active', self.colors["accent"])],
                 foreground=[('active', self.colors["text"])])
        
        # Configurar entrada de texto
        style.configure('Neon.TEntry',
                       fieldbackground=self.colors["bg_light"],
                       foreground=self.colors["text"],
                       bordercolor=self.colors["neon"],
                       focuscolor=self.colors["neon"])
        
        # Configurar LabelFrame
        style.configure('Neon.TLabelframe', 
                       background=self.colors["bg_dark"],
                       foreground=self.colors["neon"],
                       bordercolor=self.colors["neon"])
        style.configure('Neon.TLabelframe.Label', 
                       background=self.colors["bg_dark"],
                       foreground=self.colors["neon"],
                       font=('Roboto', 10, 'bold'))
        
    def crear_boton(self, parent, text, command, width=25):
        return ttk.Button(parent, text=text, command=command, style='Neon.TButton', width=width)
    
    def crear_entrada(self, parent, textvariable, width=10):
        return ttk.Entry(parent, textvariable=textvariable, style='Neon.TEntry', width=width)
    
    def crear_etiqueta(self, parent, text, style='TLabel'):
        return ttk.Label(parent, text=text, style=style)
    
    def crear_frame_estilo(self, parent, text=None):
        if text:
            return ttk.LabelFrame(parent, text=text, style='Neon.TLabelframe')
        return ttk.Frame(parent, style='TFrame')
    
    def mostrar_menu_principal(self):
        # Limpiar ventana principal
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Frame principal con degradado
        frame_principal = self.crear_frame_estilo(self.root)
        frame_principal.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título con efecto neón
        titulo = self.crear_etiqueta(frame_principal, 
                                    "SISTEMA DE GENERACIÓN DE NÚMEROS PSEUDOALEATORIOS", 
                                    'Title.TLabel')
        titulo.pack(pady=30)
        
        # Subtítulo
        subtitulo = self.crear_etiqueta(frame_principal, 
                                       "Seleccione un método para generar números pseudoaleatorios",
                                       'Neon.TLabel')
        subtitulo.pack(pady=(0, 30))
        
        # Frame para botones con diseño de cuadrícula futurista
        frame_botones = self.crear_frame_estilo(frame_principal)
        frame_botones.pack(pady=20)
        
        # Botones del menú principal con estilo neón
        botones = [
            ("Método de Cuadrados Medios", self.mostrar_cuadrados_medios),
            ("Método de Productos Medios", self.mostrar_productos_medios),
            ("Método del Multiplicador Constante", self.mostrar_multiplicador_constante),
            ("Pruebas Estadísticas", self.mostrar_pruebas_estadisticas)
        ]
        
        for texto, comando in botones:
            btn = self.crear_boton(frame_botones, texto, comando, width=30)
            btn.pack(pady=12)
        
        # Footer con estilo futurista
        footer = self.crear_etiqueta(frame_principal, 
                                    "© 2025 Sistema de Simulación ", 
                                    'Neon.TLabel')
        footer.pack(side='bottom', pady=20)
    
    # ==================== MÉTODOS DE GENERACIÓN ====================
    def mostrar_cuadrados_medios(self):
        self.limpiar_ventana()
        self.metodo_actual = "Cuadrados Medios"
        
        # Frame principal
        frame_principal = self.crear_frame_estilo(self.root)
        frame_principal.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        titulo = self.crear_etiqueta(frame_principal, "MÉTODO DE CUADRADOS MEDIOS", 'Title.TLabel')
        titulo.pack(pady=15)
        
        # Frame de parámetros
        frame_params = self.crear_frame_estilo(frame_principal, "Parámetros")
        frame_params.pack(fill='x', padx=10, pady=10)
        
        self.crear_etiqueta(frame_params, "Cantidad de números (n):").grid(row=0, column=0, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.n).grid(row=0, column=1, padx=5, pady=8)
        
        self.crear_etiqueta(frame_params, "Semilla inicial:").grid(row=0, column=2, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.semilla1_cuadrados).grid(row=0, column=3, padx=5, pady=8)
        
        # Frame para pruebas estadísticas
        frame_pruebas = self.crear_frame_estilo(frame_principal, "Pruebas Estadísticas")
        frame_pruebas.pack(fill='x', padx=10, pady=10)
        
        # Configuración de nivel de confianza para pruebas
        self.crear_etiqueta(frame_pruebas, "Nivel de Confianza:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_pruebas, self.confianza_medias).grid(row=0, column=1, padx=5, pady=5)
        
        # Botones
        frame_botones = self.crear_frame_estilo(frame_principal)
        frame_botones.pack(pady=12)
        
        self.crear_boton(frame_botones, "Generar Números", self.generar_cuadrados_medios).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Medias", self.prueba_medias).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Varianza", self.prueba_varianza).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Uniformidad", self.prueba_uniformidad).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Mostrar Histograma", self.mostrar_histograma).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Exportar a TXT", self.exportar_txt).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Atrás", self.mostrar_menu_principal).pack(side='left', padx=8)
        
        # Área de resultados
        frame_resultados = self.crear_frame_estilo(frame_principal, "Resultados")
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar área de texto con estilo
        self.text_resultados = scrolledtext.ScrolledText(
            frame_resultados, 
            height=15,
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            insertbackground=self.colors["neon"],
            selectbackground=self.colors["accent"],
            font=('Consolas', 9)
        )
        self.text_resultados.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configurar tags para colores
        self.text_resultados.tag_configure("title", foreground=self.colors["neon"])
        self.text_resultados.tag_configure("divider", foreground=self.colors["accent"])
        self.text_resultados.tag_configure("success", foreground=self.colors["highlight"])
        self.text_resultados.tag_configure("error", foreground="#ff5555")
        self.text_resultados.tag_configure("subtitle", foreground=self.colors["accent"])
    
    def mostrar_productos_medios(self):
        self.limpiar_ventana()
        self.metodo_actual = "Productos Medios"
        
        # Frame principal
        frame_principal = self.crear_frame_estilo(self.root)
        frame_principal.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        titulo = self.crear_etiqueta(frame_principal, "MÉTODO DE PRODUCTOS MEDIOS", 'Title.TLabel')
        titulo.pack(pady=15)
        
        # Frame de parámetros
        frame_params = self.crear_frame_estilo(frame_principal, "Parámetros")
        frame_params.pack(fill='x', padx=10, pady=10)
        
        self.crear_etiqueta(frame_params, "Cantidad de números (n):").grid(row=0, column=0, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.n).grid(row=0, column=1, padx=5, pady=8)
        
        self.crear_etiqueta(frame_params, "Semilla 1:").grid(row=0, column=2, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.semilla1_medios).grid(row=0, column=3, padx=5, pady=8)
        
        self.crear_etiqueta(frame_params, "Semilla 2:").grid(row=1, column=0, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.semilla2_medios).grid(row=1, column=1, padx=5, pady=8)
        
        # Frame para pruebas estadísticas
        frame_pruebas = self.crear_frame_estilo(frame_principal, "Pruebas Estadísticas")
        frame_pruebas.pack(fill='x', padx=10, pady=10)
        
        # Configuración de nivel de confianza para pruebas
        self.crear_etiqueta(frame_pruebas, "Nivel de Confianza:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_pruebas, self.confianza_medias).grid(row=0, column=1, padx=5, pady=5)
        
        # Botones
        frame_botones = self.crear_frame_estilo(frame_principal)
        frame_botones.pack(pady=12)
        
        self.crear_boton(frame_botones, "Generar Números", self.generar_productos_medios).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Medias", self.prueba_medias).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba of Varianza", self.prueba_varianza).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Uniformidad", self.prueba_uniformidad).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Mostrar Histograma", self.mostrar_histograma).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Exportar a TXT", self.exportar_txt).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Atrás", self.mostrar_menu_principal).pack(side='left', padx=8)
        
        # Área de resultados
        frame_resultados = self.crear_frame_estilo(frame_principal, "Resultados")
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.text_resultados = scrolledtext.ScrolledText(
            frame_resultados, 
            height=15,
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            insertbackground=self.colors["neon"],
            selectbackground=self.colors["accent"],
            font=('Consolas', 9)
        )
        self.text_resultados.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configurar tags para colores
        self.text_resultados.tag_configure("title", foreground=self.colors["neon"])
        self.text_resultados.tag_configure("divider", foreground=self.colors["accent"])
        self.text_resultados.tag_configure("success", foreground=self.colors["highlight"])
        self.text_resultados.tag_configure("error", foreground="#ff5555")
        self.text_resultados.tag_configure("subtitle", foreground=self.colors["accent"])
    
    def mostrar_multiplicador_constante(self):
        self.limpiar_ventana()
        self.metodo_actual = "Multiplicador Constante"
        
        # Frame principal
        frame_principal = self.crear_frame_estilo(self.root)
        frame_principal.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        titulo = self.crear_etiqueta(frame_principal, "MÉTODO DEL MULTIPLICADOR CONSTANTE", 'Title.TLabel')
        titulo.pack(pady=15)
        
        # Frame de parámetros
        frame_params = self.crear_frame_estilo(frame_principal, "Parámetros")
        frame_params.pack(fill='x', padx=10, pady=10)
        
        self.crear_etiqueta(frame_params, "Cantidad de números (n):").grid(row=0, column=0, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.n).grid(row=0, column=1, padx=5, pady=8)
        
        self.crear_etiqueta(frame_params, "Semilla:").grid(row=0, column=2, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.semilla_multiplicador).grid(row=0, column=3, padx=5, pady=8)
        
        self.crear_etiqueta(frame_params, "Constante:").grid(row=1, column=0, padx=5, pady=8, sticky='e')
        self.crear_entrada(frame_params, self.constante_multiplicador).grid(row=1, column=1, padx=5, pady=8)
        
        # Frame para pruebas estadísticas
        frame_pruebas = self.crear_frame_estilo(frame_principal, "Pruebas Estadísticas")
        frame_pruebas.pack(fill='x', padx=10, pady=10)
        
        # Configuración de nivel de confianza para pruebas
        self.crear_etiqueta(frame_pruebas, "Nivel de Confianza:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_pruebas, self.confianza_medias).grid(row=0, column=1, padx=5, pady=5)
        
        # Botones
        frame_botones = self.crear_frame_estilo(frame_principal)
        frame_botones.pack(pady=12)
        
        self.crear_boton(frame_botones, "Generar Números", self.generar_multiplicador_constante).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Medias", self.prueba_medias).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Varianza", self.prueba_varianza).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Prueba de Uniformidad", self.prueba_uniformidad).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Mostrar Histograma", self.mostrar_histograma).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Exportar a TXT", self.exportar_txt).pack(side='left', padx=8)
        self.crear_boton(frame_botones, "Atrás", self.mostrar_menu_principal).pack(side='left', padx=8)
        
        # Área de resultados
        frame_resultados = self.crear_frame_estilo(frame_principal, "Resultados")
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.text_resultados = scrolledtext.ScrolledText(
            frame_resultados, 
            height=15,
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            insertbackground=self.colors["neon"],
            selectbackground=self.colors["accent"],
            font=('Consolas', 9)
        )
        self.text_resultados.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configurar tags para colores
        self.text_resultados.tag_configure("title", foreground=self.colors["neon"])
        self.text_resultados.tag_configure("divider", foreground=self.colors["accent"])
        self.text_resultados.tag_configure("success", foreground=self.colors["highlight"])
        self.text_resultados.tag_configure("error", foreground="#ff5555")
        self.text_resultados.tag_configure("subtitle", foreground=self.colors["accent"])
    
    def mostrar_pruebas_estadisticas(self):
        self.limpiar_ventana()
        
        # Frame principal
        frame_principal = self.crear_frame_estilo(self.root)
        frame_principal.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Título
        titulo = self.crear_etiqueta(frame_principal, "PRUEBAS ESTADÍSTICAS", 'Title.TLabel')
        titulo.pack(pady=15)
        
        # Prueba de Medias
        frame_medias = self.crear_frame_estilo(frame_principal, "Prueba de Medias")
        frame_medias.pack(fill='x', padx=10, pady=8)
        
        self.crear_etiqueta(frame_medias, "Nivel de Confianza (ej. 0.95):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_medias, self.confianza_medias).grid(row=0, column=1, padx=5, pady=5)
        
        self.crear_boton(frame_medias, "Ejecutar Prueba de Medias", self.prueba_medias).grid(row=0, column=2, padx=5, pady=5)
        
        # Prueba de Varianza
        frame_varianza = self.crear_frame_estilo(frame_principal, "Prueba de Varianza")
        frame_varianza.pack(fill='x', padx=10, pady=8)
        
        self.crear_etiqueta(frame_varianza, "Nivel de Confianza (ej. 0.95):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_varianza, self.confianza_varianza).grid(row=0, column=1, padx=5, pady=5)
        
        self.crear_boton(frame_varianza, "Ejecutar Prueba de Varianza", self.prueba_varianza).grid(row=0, column=2, padx=5, pady=5)
        
        # Prueba de Uniformidad (Chi-cuadrada)
        frame_uniformidad = self.crear_frame_estilo(frame_principal, "Prueba de Uniformidad (Chi-cuadrada)")
        frame_uniformidad.pack(fill='x', padx=10, pady=8)
        
        self.crear_etiqueta(frame_uniformidad, "Número de intervalos (m):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_uniformidad, self.intervalos_chi).grid(row=0, column=1, padx=5, pady=5)
        
        self.crear_etiqueta(frame_uniformidad, "Nivel de Confianza (ej. 0.95):").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.crear_entrada(frame_uniformidad, self.confianza_uniformidad).grid(row=0, column=3, padx=5, pady=5)
        
        self.crear_boton(frame_uniformidad, "Ejecutar Prueba de Uniformidad", self.prueba_uniformidad).grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.crear_boton(frame_uniformidad, "Mostrar Histograma", self.mostrar_histograma).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        self.crear_boton(frame_uniformidad, "Exportar a TXT", self.exportar_txt).grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        
        # Botón Atrás
        frame_botones = self.crear_frame_estilo(frame_principal)
        frame_botones.pack(pady=12)
        
        self.crear_boton(frame_botones, "Atrás", self.mostrar_menu_principal).pack(pady=5)
        
        # Área de resultados
        frame_resultados = self.crear_frame_estilo(frame_principal, "Resultados")
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.text_resultados = scrolledtext.ScrolledText(
            frame_resultados, 
            height=15,
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            insertbackground=self.colors["neon"],
            selectbackground=self.colors["accent"],
            font=('Consolas', 9)
        )
        self.text_resultados.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configurar tags para colores
        self.text_resultados.tag_configure("title", foreground=self.colors["neon"])
        self.text_resultados.tag_configure("divider", foreground=self.colors["accent"])
        self.text_resultados.tag_configure("success", foreground=self.colors["highlight"])
        self.text_resultados.tag_configure("error", foreground="#ff5555")
        self.text_resultados.tag_configure("subtitle", foreground=self.colors["accent"])
    
    def limpiar_ventana(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    # ==================== GENERACIÓN DE NÚMEROS ====================
    def generar_cuadrados_medios(self):
        try:
            n = self.n.get()
            semilla = self.semilla1_cuadrados.get()
            
            self.numeros_generados, self.historial_generacion = NumberGeneration.cuadrados_medios(semilla, n)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, "GENERANDO NÚMEROS POR CUADRADOS MEDIOS\n", "title")
            self.text_resultados.insert(tk.END, "="*60 + "\n", "divider")
            
            for item in self.historial_generacion:
                self.text_resultados.insert(tk.END, 
                    f"Iteración {item['iteracion']}: {item['yi']}² = {item['yi_cuadrado']} -> {item['yi_estrella']} -> {item['ri']:.4f}\n")
            
            self.text_resultados.insert(tk.END, "="*60 + "\n", "divider")
            self.text_resultados.insert(tk.END, f"Generados {len(self.numeros_generados)} números pseudoaleatorios\n", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar números: {str(e)}")
    
    def generar_productos_medios(self):
        try:
            n = self.n.get()
            semilla1 = self.semilla1_medios.get()
            semilla2 = self.semilla2_medios.get()
            
            self.numeros_generados, self.historial_generacion = NumberGeneration.productos_medios(semilla1, semilla2, n)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, "GENERANDO NÚMEROS POR PRODUCTOS MEDIOS\n", "title")
            self.text_resultados.insert(tk.END, "="*70 + "\n", "divider")
            
            for item in self.historial_generacion:
                self.text_resultados.insert(tk.END, 
                    f"Iteración {item['iteracion']}: {item['yi0']}×{item['yi1']} = {item['producto']} -> {item['yi_estrella']} -> {item['ri']:.4f}\n")
            
            self.text_resultados.insert(tk.END, "="*70 + "\n", "divider")
            self.text_resultados.insert(tk.END, f"Generados {len(self.numeros_generados)} números pseudoaleatorios\n", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar números: {str(e)}")
    
    def generar_multiplicador_constante(self):
        try:
            n = self.n.get()
            semilla = self.semilla_multiplicador.get()
            constante = self.constante_multiplicador.get()
            
            self.numeros_generados, self.historial_generacion = NumberGeneration.multiplicador_constante(semilla, constante, n)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, "GENERANDO NÚMEROS POR MULTIPLICADOR CONSTANTE\n", "title")
            self.text_resultados.insert(tk.END, "="*70 + "\n", "divider")
            
            for item in self.historial_generacion:
                self.text_resultados.insert(tk.END, 
                    f"Iteración {item['iteracion']}: {item['constante']}×{item['yi']} = {item['producto']} -> {item['yi_estrella']} -> {item['ri']:.4f}\n")
            
            self.text_resultados.insert(tk.END, "="*70 + "\n", "divider")
            self.text_resultados.insert(tk.END, f"Generados {len(self.numeros_generados)} números pseudoaleatorios\n", "success")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar números: {str(e)}")
    
    # ==================== PRUEBAS ESTADÍSTICAS ====================
    def prueba_medias(self):
        if not self.numeros_generados:
            messagebox.showwarning("Advertencia", "Primero genere números aleatorios")
            return
        
        try:
            confianza = self.confianza_medias.get()
            media, li, ls, z_alpha, pasa_prueba = StatisticalTests.media_test(self.numeros_generados, confianza)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, f"=== PRUEBA DE MEDIAS ({self.metodo_actual}) ===\n", "title")
            self.text_resultados.insert(tk.END, f"{'Cantidad de números:':<25} {len(self.numeros_generados):>10}\n")
            self.text_resultados.insert(tk.END, f"{'Media calculada:':<25} {media:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Límite inferior:':<25} {li:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Límite superior:':<25} {ls:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Valor Z_alpha:':<25} {z_alpha:.4f}\n")
            self.text_resultados.insert(tk.END, f"{'Nivel de confianza:':<25} {confianza}\n")
            
            if pasa_prueba:
                self.text_resultados.insert(tk.END, "✅ CONCLUSIÓN: Los números pasan la prueba de medias\n", "success")
            else:
                self.text_resultados.insert(tk.END, "❌ CONCLUSIÓN: Los números NO pasan la prueba de medias\n", "error")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en prueba de medias: {str(e)}")
    
    def prueba_varianza(self):
        if not self.numeros_generados:
            messagebox.showwarning("Advertencia", "Primero genere números aleatorios")
            return
        
        try:
            confianza = self.confianza_varianza.get()
            varianza, li, ls, chi2_inf, chi2_sup, pasa_prueba = StatisticalTests.varianza_test(self.numeros_generados, confianza)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, f"=== PRUEBA DE VARIANZA ({self.metodo_actual}) ===\n", "title")
            self.text_resultados.insert(tk.END, f"{'Cantidad de números:':<25} {len(self.numeros_generados):>10}\n")
            self.text_resultados.insert(tk.END, f"{'Varianza calculada:':<25} {varianza:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Límite inferior:':<25} {li:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Límite superior:':<25} {ls:.6f}\n")
            self.text_resultados.insert(tk.END, f"{'Chi² inferior:':<25} {chi2_inf:.4f}\n")
            self.text_resultados.insert(tk.END, f"{'Chi² superior:':<25} {chi2_sup:.4f}\n")
            self.text_resultados.insert(tk.END, f"{'Nivel de confianza:':<25} {confianza}\n")
            
            if pasa_prueba:
                self.text_resultados.insert(tk.END, "✅ CONCLUSIÓN: Los números pasan la prueba de varianza\n", "success")
            else:
                self.text_resultados.insert(tk.END, "❌ CONCLUSIÓN: Los números NO pasan la prueba de varianza\n", "error")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en prueba de varianza: {str(e)}")
    
    def prueba_uniformidad(self):
        if not self.numeros_generados:
            messagebox.showwarning("Advertencia", "Primero genere números aleatorios")
            return
        
        try:
            intervalos = self.intervalos_chi.get()
            confianza = self.confianza_uniformidad.get()
            frec_obs, frec_esp, chi2_calculado, chi2_critico, gl, bins, pasa_prueba = StatisticalTests.uniformidad_test(
                self.numeros_generados, intervalos, confianza)
            
            self.text_resultados.delete(1.0, tk.END)
            self.text_resultados.insert(tk.END, f"=== PRUEBA DE UNIFORMIDAD (CHI-CUADRADA) ({self.metodo_actual}) ===\n", "title")
            self.text_resultados.insert(tk.END, f"{'Cantidad de números:':<25} {len(self.numeros_generados):>10}\n")
            self.text_resultados.insert(tk.END, f"{'Número de intervalos:':<25} {intervalos:>10}\n")
            self.text_resultados.insert(tk.END, f"{'Frecuencia esperada:':<25} {frec_esp:.2f}\n")
            self.text_resultados.insert(tk.END, f"{'Chi-cuadrado calculado:':<25} {chi2_calculado:.4f}\n")
            self.text_resultados.insert(tk.END, f"{'Chi-cuadrado crítico:':<25} {chi2_critico:.4f}\n")
            self.text_resultados.insert(tk.END, f"{'Grados de libertad:':<25} {gl:>10}\n")
            self.text_resultados.insert(tk.END, f"{'Nivel de confianza:':<25} {confianza}\n")
                        # Mostrar tabla de frecuencias
            self.text_resultados.insert(tk.END, "\nTABLA DE FRECUENCIAS:\n", "subtitle")
            self.text_resultados.insert(tk.END, f"{'Intervalo':<15} {'Frec. Observada':<15} {'Frec. Esperada':<15} {'Diferencia':<15}\n")
            self.text_resultados.insert(tk.END, "-"*60 + "\n")
            
            for i in range(intervalos):
                intervalo = f"[{bins[i]:.2f}-{bins[i+1]:.2f})"
                diferencia = (frec_obs[i] - frec_esp)**2 / frec_esp
                self.text_resultados.insert(tk.END, f"{intervalo:<15} {frec_obs[i]:<15.0f} {frec_esp:<15.2f} {diferencia:<15.4f}\n")
            
            if pasa_prueba:
                self.text_resultados.insert(tk.END, "✅ CONCLUSIÓN: Los números pasan la prueba de uniformidad\n", "success")
            else:
                self.text_resultados.insert(tk.END, "❌ CONCLUSIÓN: Los números NO pasan la prueba de uniformidad\n", "error")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en prueba de uniformidad: {str(e)}")
    
    def mostrar_histograma(self):
        if not self.numeros_generados:
            messagebox.showwarning("Advertencia", "Primero genere números aleatorios")
            return
        
        try:
            # Crear una nueva ventana para el histograma
            ventana_hist = tk.Toplevel(self.root)
            ventana_hist.title("Histograma de Números Generados")
            ventana_hist.geometry("900x700")
            ventana_hist.configure(bg=self.colors["bg_dark"])
            
            # Crear figura de matplotlib
            fig = Figure(figsize=(9, 7), dpi=100)
            ax = fig.add_subplot(111)
            
            # Generar histograma en forma de pirámide
            n_bins = min(self.intervalos_chi.get(), 20)  # Limitar a 20 bins máximo para mejor visualización
            n, bins, patches = ax.hist(self.numeros_generados, bins=n_bins, color=self.colors["accent"], 
                   edgecolor=self.colors["neon"], alpha=0.7, density=True)
            
            # Calcular frecuencia esperada (0.5 para distribución uniforme)
            freq_esperada = 0.5
            
            # Añadir línea de frecuencia esperada que recorra todo el histograma
            x_vals = np.linspace(0, 1, 100)
            y_vals = np.full_like(x_vals, freq_esperada)
            ax.plot(x_vals, y_vals, color=self.colors["highlight"], linestyle='--', 
                   linewidth=2, label='Frecuencia Esperada (0.5)')
            
            # Añadir línea que sigue la forma de pirámide del histograma
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(bin_centers, n, color=self.colors["neon"], linestyle='-', 
                   linewidth=2, marker='o', markersize=4, label='Frecuencia Observada')
            
            # Personalizar el gráfico
            ax.set_title('Distribución de Números Pseudoaleatorios', 
                        color=self.colors["text"], fontsize=14, pad=20)
            ax.set_xlabel('Valor', color=self.colors["text"], fontsize=12)
            ax.set_ylabel('Densidad de Frecuencia', color=self.colors["text"], fontsize=12)
            ax.tick_params(colors=self.colors["text"])
            
            # Cambiar color del fondo
            ax.set_facecolor(self.colors["bg_light"])
            fig.patch.set_facecolor(self.colors["bg_dark"])
            
            # Añadir cuadrícula
            ax.grid(True, alpha=0.3, color=self.colors["neon"])
            
            # Añadir leyenda
            ax.legend()
            
            # Integrar matplotlib con tkinter
            canvas = FigureCanvasTkAgg(fig, master=ventana_hist)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Añadir botón de cerrar
            btn_cerrar = self.crear_boton(ventana_hist, "Cerrar", ventana_hist.destroy)
            btn_cerrar.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al mostrar histograma: {str(e)}")
    
    # ==================== EXPORTACIÓN A TXT ====================
    def exportar_txt(self):
        if not self.numeros_generados:
            messagebox.showwarning("Advertencia", "Primero genere números aleatorios")
            return
        
        try:
            # Obtener el contenido del área de texto
            contenido = self.text_resultados.get(1.0, tk.END)
            
            # Solicitar al usuario dónde guardar el archivo
            archivo = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                title="Guardar resultados como"
            )
            
            if archivo:
                # Guardar el contenido en el archivo
                with open(archivo, 'w', encoding='utf-8') as f:
                    f.write(f"Resultados de {self.metodo_actual}\n")
                    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*50 + "\n")
                    f.write(contenido)
                
                messagebox.showinfo("Éxito", f"Resultados exportados correctamente a:\n{archivo}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar a TXT: {str(e)}")

# ==================== FUNCIÓN PRINCIPAL ====================
def main():
    root = tk.Tk()
    app = RandomNumberApp(root)
    root.mainloop()

if __name__ == "__main__":

    main()
