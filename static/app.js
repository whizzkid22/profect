/* ═══════════════════════════════════════════════
   Hair Vision — app.js (полное отображение ML)
═══════════════════════════════════════════════ */

// 1. Переключение табов
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('tab--active'));
        document.querySelectorAll('.pane').forEach(p => p.classList.remove('pane--active'));
        tab.classList.add('tab--active');
        const targetPane = document.getElementById('pane-' + tab.dataset.tab);
        if (targetPane) targetPane.classList.add('pane--active');
    });
});

// Вспомогательная функция для чистки Base64
function formatBase64Image(rawStr) {
    if (!rawStr) return null;
    const cleanStr = rawStr.trim().replace(/\s/g, '').replace(/['"]/g, '');
    if (cleanStr.startsWith('data:image')) return cleanStr;
    return `data:image/jpeg;base64,${cleanStr}`;
}

// 2. Функция проверки кнопки HairFast
function checkShBtn() {
    const ids = ['shFaceFile', 'shShapeFile', 'shColorFile'];
    const allLoaded = ids.every(id => {
        const el = document.getElementById(id);
        return el && (el._file || (el.files && el.files[0]));
    });
    const mainBtn = document.getElementById('shBtn');
    if (mainBtn) mainBtn.disabled = !allLoaded;
}

// 3. Фабрика зон загрузки
function setupUploadZone(zoneId, fileId, previewId, btnId) {
    const zone    = document.getElementById(zoneId);
    const input   = document.getElementById(fileId);
    const preview = document.getElementById(previewId);
    const btn     = btnId ? document.getElementById(btnId) : null;
    if (!zone || !input) return;

    function loadFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        input._file = file;
        const url = URL.createObjectURL(file);
        if (preview) {
            preview.src = url;
            preview.style.display = 'block';
            preview.onload = () => zone.classList.add('upload-zone--has-img');
        }
        if (btn) btn.disabled = false;
        checkShBtn();
    }

    zone.addEventListener('click', (e) => { if (e.target !== input) input.click(); });
    input.addEventListener('change', () => { if (input.files.length > 0) loadFile(input.files[0]); });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; });
    zone.addEventListener('dragleave', () => zone.style.borderColor = '');
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.style.borderColor = '';
        if (e.dataTransfer.files.length > 0) loadFile(e.dataTransfer.files[0]);
    });
}

// 4. Инициализация всех зон
setupUploadZone('mlZone',     'mlFile',     'mlPreview',     'mlBtn');
setupUploadZone('llmZone',    'llmFile',    'llmPreview',    'llmBtn');
setupUploadZone('shFaceZone', 'shFaceFile', 'shFacePreview', null);
setupUploadZone('shShapeZone','shShapeFile','shShapePreview',null);
setupUploadZone('shColorZone','shColorFile','shColorPreview',null);

// 5. Рисование формы лица
const SHAPE_LABELS_RU = {
    oval: 'Овальное', round: 'Круглое', square: 'Квадратное',
    heart: 'Сердцевидное', oblong: 'Продолговатое'
};

function drawFaceShape(canvas, shape) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#c8a96e';
    ctx.lineWidth = 2.5;
    ctx.fillStyle = 'rgba(200,169,110,0.1)';
    ctx.beginPath();
    if (shape === 'round')       ctx.arc(cx, cy, Math.min(W, H) * 0.38, 0, Math.PI * 2);
    else if (shape === 'square') ctx.rect(cx - 35, cy - 35, 70, 70);
    else if (shape === 'oblong') ctx.ellipse(cx, cy, W * 0.26, H * 0.46, 0, 0, Math.PI * 2);
    else                         ctx.ellipse(cx, cy, W * 0.32, H * 0.43, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
}

// 6. HairFastGAN
const shBtn = document.getElementById('shBtn');
if (shBtn) {
    shBtn.addEventListener('click', async () => {
        const faceF  = document.getElementById('shFaceFile')._file  || document.getElementById('shFaceFile').files[0];
        const shapeF = document.getElementById('shShapeFile')._file || document.getElementById('shShapeFile').files[0];
        const colorF = document.getElementById('shColorFile')._file || document.getElementById('shColorFile').files[0];

        const loader   = document.getElementById('shLoader');
        const resultEl = document.getElementById('shResult');
        const errBox   = document.getElementById('shError');

        if (loader) loader.hidden = false;
        if (errBox) errBox.hidden = true;
        shBtn.disabled = true;

        try {
            const fd = new FormData();
            fd.append('face_file',  faceF);
            fd.append('shape_file', shapeF);
            fd.append('color_file', colorF);

            const resp = await fetch('/api/hair-transfer', { method: 'POST', body: fd });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Ошибка сервера');

            // Перебираем возможные ключи с изображением
            const rawImg = data.result_image ?? data.result ?? data.image ??
                           data.output ?? data.output_image ?? data.generated ?? null;
            const finalImgData = formatBase64Image(rawImg);

            if (!finalImgData) {
                console.error('Ответ воркера:', data);
                throw new Error(`Сервер не вернул изображение. Ключи ответа: ${Object.keys(data).join(', ')}`);
            }

            document.getElementById('shOrigImg').src  = URL.createObjectURL(faceF);
            document.getElementById('shShapeImg').src = URL.createObjectURL(shapeF);
            document.getElementById('shColorImg').src = URL.createObjectURL(colorF);
            document.getElementById('shResultImg').src = finalImgData;

            const dlBtn = document.getElementById('shDownloadBtn');
            if (dlBtn) {
                dlBtn.onclick = () => {
                    const a = document.createElement('a');
                    a.href = finalImgData;
                    a.download = 'hair_vision_result.jpg';
                    a.click();
                };
            }
            if (resultEl) resultEl.hidden = false;
        } catch (e) {
            console.error('HairTransfer Error:', e);
            if (errBox) { errBox.textContent = e.message; errBox.hidden = false; }
            else alert(e.message);
        } finally {
            if (loader) loader.hidden = true;
            shBtn.disabled = false;
        }
    });
}

// 7. ML-анализ — ПОЛНОЕ заполнение всех полей
const mlBtn = document.getElementById('mlBtn');
if (mlBtn) {
    mlBtn.addEventListener('click', async () => {
        const file   = document.getElementById('mlFile')._file || document.getElementById('mlFile').files[0];
        const loader = document.getElementById('mlLoader');
        const result = document.getElementById('mlResult');
        const errBox = document.getElementById('mlError');

        if (loader) loader.hidden = false;
        if (result) result.hidden = true;
        if (errBox) errBox.hidden = true;

        try {
            const fd = new FormData();
            fd.append('file', file);
            const resp = await fetch('/api/ml-analyze', { method: 'POST', body: fd });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Ошибка ML');

            // ── Тип волос ──────────────────────────────────────────────────
            document.getElementById('mlHairType').textContent = data.hair_type || '—';

            const conf = Math.round(data.confidence || 0);
            document.getElementById('mlConf').textContent = conf + '%';
            const bar = document.getElementById('mlConfBar');
            if (bar) bar.style.width = conf + '%';

            const hairTip = document.getElementById('mlHairTip');
            if (hairTip) hairTip.textContent = data.hair_type_tip || '';

            // ── Форма лица ────────────────────────────────────────────────
            const shape = data.face_shape || 'oval';
            drawFaceShape(document.getElementById('faceShapeCanvas'), shape);
            document.getElementById('mlFaceShapeName').textContent = SHAPE_LABELS_RU[shape] || shape;

            const faceTip = document.getElementById('mlFaceTip');
            if (faceTip) faceTip.textContent = data.face_shape_tip || '';

            // ── Персона: пол и возраст ────────────────────────────────────
            const genderEl = document.getElementById('mlGender');
            if (genderEl) {
                const gMap = { Man: 'Мужчина', Woman: 'Женщина', Unknown: 'Неизвестно' };
                const gLabel = gMap[data.gender] || data.gender || '—';
                const gConf = data.gender_conf ? ` (${data.gender_conf}%)` : '';
                genderEl.textContent = gLabel + gConf;
            }

            const ageEl = document.getElementById('mlAge');
            if (ageEl) {
                const ageGroupMap = { young: 'молодой', middle: 'средний возраст', senior: 'зрелый' };
                const ageLabel = data.age > 0 ? data.age + ' лет' : '—';
                const ageGroup = data.age_group ? ` · ${ageGroupMap[data.age_group] || data.age_group}` : '';
                ageEl.textContent = ageLabel + ageGroup;
            }

            // ── Топ-3 прогноза ────────────────────────────────────────────
            const top3El = document.getElementById('mlTop3');
            if (top3El && Array.isArray(data.top3)) {
                top3El.innerHTML = data.top3.map((item, i) => `
                    <div class="top3__item ${i === 0 ? 'top3__item--best' : ''}">
                        <span class="top3__label">${item.label}</span>
                        <span class="top3__bar-wrap">
                            <span class="top3__bar" style="width:${item.prob}%"></span>
                        </span>
                        <span class="top3__prob">${item.prob}%</span>
                    </div>`).join('');
            }

            // ── Геометрия лица ────────────────────────────────────────────
            const geoEl = document.getElementById('mlGeo');
            if (geoEl && data.geo) {
                const geoLabels = {
                    face_ratio:             'Соотношение В/Ш',
                    forehead_to_face_width: 'Лоб / ширина лица',
                    jaw_to_face_width:      'Челюсть / ширина лица',
                    cheek_to_face_width:    'Скулы / ширина лица',
                    eye_distance_ratio:     'Расстояние между глаз',
                    nose_width_to_face:     'Нос / ширина лица',
                    mouth_to_face_width:    'Рот / ширина лица',
                };
                geoEl.innerHTML = Object.entries(data.geo).map(([key, val]) => `
                    <div class="geo-bar">
                        <span class="geo-bar__label">${geoLabels[key] || key}</span>
                        <span class="geo-bar__val">${val}</span>
                    </div>`).join('');
            }

            // ── Веса ансамбля ─────────────────────────────────────────────
            const wEl = document.getElementById('mlWeights');
            if (wEl && data.model_weights) {
                const wLabels = { cnn: 'CNN', xgb: 'XGBoost', lgb: 'LightGBM', gb: 'GradBoost' };
                wEl.innerHTML = Object.entries(data.model_weights).map(([k, v]) => `
                    <div class="weight-row">
                        <span class="weight-row__name">${wLabels[k] || k}</span>
                        <span class="weight-row__bar-wrap">
                            <span class="weight-row__bar" style="width:${Math.round(v * 100)}%"></span>
                        </span>
                        <span class="weight-row__val">${Math.round(v * 100)}%</span>
                    </div>`).join('');
            }

            // ── Симуляция лысины ──────────────────────────────────────────
        const baldImgData = formatBase64Image(data.bald_image);
        let baldBox = document.getElementById('mlBaldSection');
        if (!baldBox) {
            baldBox = document.createElement('div');
            baldBox.id = 'mlBaldSection';
            baldBox.className = 'result-card';
            baldBox.style.gridColumn = '1 / -1';
            baldBox.innerHTML = `
                <div class="result-card__label">Симуляция лысины</div>
                <div style="display:flex;gap:16px;align-items:flex-start;">
                    <img id="mlBaldImg" style="max-width:220px;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.3)" alt="симуляция лысины">
                </div>`;
            result.appendChild(baldBox);
        }
        // Показываем/скрываем секцию в зависимости от наличия данных
        baldBox.hidden = !baldImgData;
        if (baldImgData) {
            document.getElementById('mlBaldImg').src = baldImgData;
        }

            if (result) result.hidden = false;
        } catch (e) {
            console.error('ML Error:', e);
            if (errBox) { errBox.textContent = e.message; errBox.hidden = false; }
        } finally {
            if (loader) loader.hidden = true;
        }
    });
}

// 8. LLM-стилист
const llmBtn = document.getElementById('llmBtn');
if (llmBtn) {
    llmBtn.addEventListener('click', async () => {
        const file   = document.getElementById('llmFile')._file || document.getElementById('llmFile').files[0];
        const prefs  = document.getElementById('llmPrefs').value;
        const loader = document.getElementById('llmLoader');
        const result = document.getElementById('llmResult');
        const errBox = document.getElementById('llmError');

        if (loader) loader.hidden = false;
        if (result) result.hidden = true;
        if (errBox) errBox.hidden = true;

        try {
            const fd = new FormData();
            fd.append('file', file);
            fd.append('prefs', prefs);
            const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
            const data = await resp.json();
	    if (!resp.ok) throw new Error(data.detail || 'Ошибка сервера'); 

            document.getElementById('llmDescription').textContent = data.description || 'Нет описания';
            document.getElementById('llmRaw').textContent = JSON.stringify(data, null, 2);

            // ── LLM атрибуты (форма лица, тип волос, рекомендации) ────────
            const attrsEl = document.getElementById('llmAttrs');
            if (attrsEl) {
                const sections = [
                    { label: 'Форма лица',    value: data.face_shape },
                    { label: 'Тип волос',     value: data.hair_type },
                    { label: 'Текущий стиль', value: data.current_style },
                ];
                const lists = [
                    { label: 'Рекомендуемые стрижки', items: data.recommended_cuts },
                    { label: 'Рекомендуемые цвета',   items: data.recommended_colors },
                    { label: 'Чего избегать',          items: data.avoid },
                    { label: 'Уход за волосами',       items: data.care_tips },
                ];

                attrsEl.innerHTML =
                    sections.filter(s => s.value).map(s => `
                        <div class="llm-attr">
                            <span class="llm-attr__label">${s.label}:</span>
                            <span class="llm-attr__val">${s.value}</span>
                        </div>`).join('') +
                    lists.filter(l => Array.isArray(l.items) && l.items.length).map(l => `
                        <div class="llm-list">
                            <div class="llm-list__label">${l.label}</div>
                            <ul class="llm-list__items">
                                ${l.items.map(i => `<li>${i}</li>`).join('')}
                            </ul>
                        </div>`).join('');
            }

            if (result) result.hidden = false;
        } catch (e) {
            console.error('LLM Error:', e);
            if (errBox) { errBox.textContent = e.message; errBox.hidden = false; }
            else alert('LLM Error: ' + e.message);
        } finally {
            if (loader) loader.hidden = true;
        }
    });
}