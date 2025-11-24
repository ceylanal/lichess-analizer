import streamlit as st
import berserk
import chess.pgn
import chess.svg
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go
import base64
import numpy as np
import google.generativeai as genai

# --- 1. AYARLAR VE CSS ---
st.set_page_config(page_title="Strategy Core v8.1", layout="wide", page_icon="♟️")

def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap');
        
        /* Genel Sayfa Yapısı */
        .stApp {
            background-color: #05080e;
            background-image: radial-gradient(#111928 1px, transparent 1px);
            background-size: 20px 20px;
            color: #b0c4de;
            font-family: 'Roboto', sans-serif;
        }
        
        /* Başlıklar */
        h1, h2, h3, .big-font {
            font-family: 'Orbitron', sans-serif !important;
            color: #00e5ff !important;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.7);
        }

        /* Sohbet Balonları */
        .stChatMessage {
            background-color: rgba(15, 22, 35, 0.8);
            border: 1px solid #1f2937;
            border-radius: 10px;
        }
        
        /* Giriş Kartı */
        .login-card {
            background-color: rgba(15, 22, 35, 0.9);
            border: 2px solid #00e5ff;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 0 30px rgba(0, 229, 255, 0.2);
            text-align: center;
        }

        /* Butonlar */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #008fcc, #00e5ff);
            border: none;
            color: #000;
            font-family: 'Orbitron';
            font-weight: bold;
            padding: 10px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.8);
            transform: scale(1.02);
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. STATE YÖNETİMİ ---
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_game_context' not in st.session_state:
    st.session_state.current_game_context = None

# --- 3. LICHESS VERİ ÇEKME ---
class LichessLoader:
    def __init__(self, token=None):
        session = berserk.TokenSession(token) if token else None
        self.client = berserk.Client(session)

    def get_user_games(self, username, max_games=50):
        try:
            games = list(self.client.games.export_by_player(
                username, 
                max=int(max_games), 
                as_pgn=True, 
                evals=False, 
                opening=True
            ))
            return games
        except Exception as e:
            st.error(f"Veri çekme hatası: {e}")
            return None

# --- 4. SATRANÇ ANALİZ MOTORU ---
class ChessAnalyzer:
    def parse_games(self, pgn_list, username):
        data = []
        heatmap_data = np.zeros(64)
        
        progress_bar = st.progress(0, text="ANALYZING BATTLE DATA...")
        
        for i, pgn in enumerate(pgn_list):
            pgn_io = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_io)
            
            if game:
                headers = game.headers
                white = headers.get("White", "?")
                black = headers.get("Black", "?")
                
                # Renk ve Rakip Belirleme (Genişletilmiş)
                if white.lower() == username.lower():
                    user_color = "White"
                    opponent = black
                    user_rating = headers.get("WhiteElo", "0")
                    opp_rating = headers.get("BlackElo", "0")
                elif black.lower() == username.lower():
                    user_color = "Black"
                    opponent = white
                    user_rating = headers.get("BlackElo", "0")
                    opp_rating = headers.get("WhiteElo", "0")
                else:
                    user_color = "White"
                    opponent = "Unknown"
                    user_rating = "0"
                    opp_rating = "0"

                # Rating Dönüşümü
                try: user_rating = int(user_rating)
                except: user_rating = None
                try: opp_rating = int(opp_rating)
                except: opp_rating = None

                # Sonuç Belirleme
                result = headers.get("Result")
                status = "Draw"
                if result == "1-0":
                    status = "Win" if user_color == "White" else "Loss"
                elif result == "0-1":
                    status = "Win" if user_color == "Black" else "Loss"

                # Hamle Analizi ve Heatmap
                move_count = 0
                temp_board = game.board()
                for move in game.mainline_moves():
                    move_count += 1
                    # Isı haritası için sadece bizim hamleler
                    if temp_board.turn == (chess.WHITE if user_color == "White" else chess.BLACK):
                        heatmap_data[move.to_square] += 1
                    temp_board.push(move)

                data.append({
                    "Game_ID": f"G-{i+1:02d}",
                    "Date": headers.get("Date"),
                    "Opponent": opponent,
                    "Opp_Rating": opp_rating,
                    "User_Rating": user_rating,
                    "Color": user_color,
                    "Opening": headers.get("Opening", "Unknown").split(":")[0],
                    "Status": status,
                    "Moves": move_count,
                    "PGN": pgn,
                    "Link": headers.get("Site", "")
                })
            
            progress_bar.progress((i + 1) / len(pgn_list))
        
        progress_bar.empty()
        return pd.DataFrame(data), heatmap_data

    def calculate_material_balance(self, pgn_text, user_color):
        pgn_io = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn_io)
        board = game.board()
        
        balance_history = []
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        moves = []
        for move in game.mainline_moves():
            board.push(move)
            
            white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
            black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
            
            diff = white_mat - black_mat
            if user_color == "Black":
                diff = -diff
                
            balance_history.append(diff)
            moves.append(len(balance_history))
            
        return moves, balance_history
# --- 5. GEMINI AI KOÇ (GÜNCELLENMİŞ MODEL) ---
class GeminiCoach:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # DÜZELTME: En yeni ve hızlı model 'gemini-1.5-flash' kullanıyoruz.
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Model Hatası: {e}")
    
    def generate_narrative_report(self, game_row, balance_history):
        # Yedek (Offline) Rapor
        report = f"**Maç Özeti:** {game_row['Opening']} açılışı oynadın. "
        if game_row['Status'] == 'Win': report += "Zorlu mücadeleyi kazandın. "
        elif game_row['Status'] == 'Loss': report += "Ne yazık ki kaybettin. "
        
        if self.model:
            try:
                # Veri optimizasyonu
                balance_summary = balance_history[::10] if len(balance_history) > 20 else balance_history
                
                prompt = f"""
                Sen bir satranç koçusun. Şu maçı analiz et:
                - Oyuncu: {game_row['Color']}
                - Açılış: {game_row['Opening']}
                - Sonuç: {game_row['Status']}
                - Materyal Grafiği (Pozitif=Oyuncu Üstün): {balance_summary}
                
                Lütfen şu formatta Türkçe, kısa ve esprili bir özet geç:
                1. Nasıl başladık?
                2. Kırılma anı.
                3. Sonuç.
                """
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return report + f"\n(AI Bağlantı Hatası: {e})"
        
        return report + " (Detaylı analiz için Gemini API Key gereklidir.)"

    def get_chat_response(self, user_msg, game_context, balance_history):
        if self.model:
            try:
                prompt = f"""
                Sen Strategy Core AI koçusun.
                Oyun: {game_context['Opening']}, Sonuç: {game_context['Status']}.
                Kullanıcı sorusu: {user_msg}
                Kısa cevap ver.
                """
                response = self.model.generate_content(prompt)
                return response.text
            except:
                return "Bağlantı hatası."
        return "API Key gerekli."
# --- 6. UI YARDIMCILARI ---
def render_svg(svg):
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f"""
    <div style="border: 2px solid #00e5ff; border-radius: 10px; padding: 10px; 
    box-shadow: 0 0 20px rgba(0,229,255,0.3); display: inline-block; background: rgba(0,0,0,0.5);">
        <img src="data:image/svg+xml;base64,{b64}" width="100%" style="max-width: 450px;"/>
    </div>
    """
    st.write(html, unsafe_allow_html=True)

def update_plot_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto", color="#b0c4de"),
        title_font=dict(family="Orbitron", color="#00e5ff", size=18)
    )
    return fig

# --- 7. SAYFALAR ---
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="login-card">
            <h1 style="font-size: 3rem;">CHESS ANALYZER v8.1</h1>
            <p style="color: #b0c4de;">An AI experiement by ASC</p>
            <hr style="border-color: #00e5ff; margin: 20px 0;">
        """, unsafe_allow_html=True)
        
        username = st.text_input("Kullanıcı Adı", placeholder="Lichess ID (Örn: MagnusCarlsen)")
        
        with st.expander("💎 Gemini API Key (Ücretsiz)"):
            st.info("Google Gemini API tamamen ücretsizdir.")
            gemini_key = st.text_input("Gemini API Key", type="password")
            st.markdown("[👉 Buradan Ücretsiz Al](https://aistudio.google.com/app/apikey)")
            lichess_token = st.text_input("Lichess Token (Opsiyonel)", type="password")

        if st.button("SİSTEMİ BAŞLAT 🚀"):
            if username:
                st.session_state.user_data = {
                    'username': username, 
                    'lichess_token': lichess_token, 
                    'gemini_key': gemini_key
                }
                st.session_state.page = 'dashboard'
                st.rerun()
            else:
                st.warning("Lütfen kullanıcı adı giriniz.")
        
        st.markdown("</div>", unsafe_allow_html=True)

def dashboard_page():
    user = st.session_state.user_data
    analyzer = ChessAnalyzer()
    coach = GeminiCoach(api_key=user['gemini_key'])
    
    # Sidebar
    with st.sidebar:
        st.header(f"👤 {user['username']}")
        if st.button("⬅️ Çıkış Yap"):
            st.session_state.page = 'login'
            st.session_state.chat_history = []
            st.rerun()
            
    # Veri Yükleme
    loader = LichessLoader(token=user['lichess_token'])
    with st.spinner("Veriler işleniyor..."):
        raw_pgns = loader.get_user_games(user['username'], max_games=30)
    
    if raw_pgns:
        df, heatmap_data = analyzer.parse_games(raw_pgns, user['username'])
        
        st.title(f"MISSION REPORT: {user['username']}")
        
        tabs = st.tabs(["📊 DASHBOARD", "🗺️ ISI HARİTASI", "🛡️ AÇILIŞLAR", "⚖️ MATERYAL", "💬 GEMINI KOÇ"])

        # Tab 1: Dashboard
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Kazanma", f"%{(len(df[df['Status']=='Win'])/len(df))*100:.1f}")
            c2.metric("Toplam", len(df))
            c3.metric("Ort. Hamle", int(df['Moves'].mean()))
            c4.metric("Sonuç", df['Status'].mode()[0])
            
            st.plotly_chart(update_plot_theme(px.pie(df, names='Status', title="Sonuç Dağılımı", 
                color='Status', color_discrete_map={'Win':'#00ff9d', 'Loss':'#ff0055', 'Draw':'#00e5ff'})), use_container_width=True)

        # Tab 2: Heatmap
        with tabs[1]:
            st.plotly_chart(update_plot_theme(px.imshow(heatmap_data.reshape(8, 8)[::-1], 
                color_continuous_scale='Viridis', title="Taş Aktivite Haritası")), use_container_width=True)

        # Tab 3: Açılışlar
        with tabs[2]:
            op_stats = df.groupby('Opening')['Status'].value_counts(normalize=True).unstack(fill_value=0)
            if 'Win' in op_stats.columns:
                op_stats['Win'] *= 100
                st.plotly_chart(update_plot_theme(px.bar(op_stats.sort_values('Win', ascending=False).head(10).reset_index(), 
                    x='Win', y='Opening', orientation='h', title="En İyi Açılışlar")), use_container_width=True)

        # Tab 4: Materyal
        with tabs[3]:
            game_opts = [f"{row['Game_ID']} | {row['Opening']} ({row['Status']})" for i, row in df.iterrows()]
            sel_game_str = st.selectbox("Maç Seç:", game_opts)
            sel_game = df.iloc[game_opts.index(sel_game_str)]
            
            moves, balance = analyzer.calculate_material_balance(sel_game['PGN'], sel_game['Color'])
            
            fig = px.line(x=moves, y=balance, title="Materyal Dengesi (Yukarı = Üstünlük)")
            fig.update_traces(line_color="#00e5ff")
            st.plotly_chart(update_plot_theme(fig), use_container_width=True)

        # Tab 5: Gemini Chat
        with tabs[4]:
            col_l, col_r = st.columns([1, 2])
            
            sel_game = df.iloc[game_opts.index(sel_game_str)]
            moves, balance = analyzer.calculate_material_balance(sel_game['PGN'], sel_game['Color'])

            # Sohbet Başlatma (Maç değiştiyse)
            if st.session_state.current_game_context != sel_game['Game_ID']:
                st.session_state.current_game_context = sel_game['Game_ID']
                st.session_state.chat_history = []
                
                with st.spinner("Gemini maçı inceliyor..."):
                    intro = coach.generate_narrative_report(sel_game, balance)
                    st.session_state.chat_history.append({"role": "assistant", "content": intro})

            with col_l:
                pgn_io = io.StringIO(sel_game['PGN'])
                game = chess.pgn.read_game(pgn_io)
                board = game.end().board()
                render_svg(chess.svg.board(board=board))
                st.info(f"Açılış: {sel_game['Opening']}")
            
            with col_r:
                chat_container = st.container(height=400)
                for msg in st.session_state.chat_history:
                    chat_container.chat_message(msg["role"]).write(msg["content"])
                
                if prompt := st.chat_input("Koça soru sor..."):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    chat_container.chat_message("user").write(prompt)
                    
                    with st.spinner("Cevap yazılıyor..."):
                        reply = coach.get_chat_response(prompt, sel_game, balance)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        chat_container.chat_message("assistant").write(reply)

    else:
        st.error("Oyun verisi bulunamadı.")

# --- 8. UYGULAMA BAŞLATMA ---
if st.session_state.page == 'login':
    login_page()
else:
    dashboard_page()