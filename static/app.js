let members=[], events=[], inside=[], config={}, cameraConfig={}, discovered=[], editingMemberId=null;
const $=id=>document.getElementById(id);
function toast(msg,error=false){const t=$('toast');t.textContent=msg;t.className='toast show'+(error?' error':'');setTimeout(()=>t.className='toast',4000)}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function activateVideoFeeds(page){const feeds={dashboard:{dashboardEntryFeed:'/video_feed/entry',dashboardExitFeed:'/video_feed/exit'},monitoring:{monitorEntryFeed:'/video_feed/entry',monitorExitFeed:'/video_feed/exit'}};const active=feeds[page]||{};['dashboardEntryFeed','dashboardExitFeed','monitorEntryFeed','monitorExitFeed'].forEach(id=>{const el=$(id);if(!el)return;const src=active[id];if(src){if(el.dataset.stream!==src){el.src=src+'?t='+Date.now();el.dataset.stream=src}}else{el.removeAttribute('src');delete el.dataset.stream}})}
function showPage(page){activateVideoFeeds(page);document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));$('page-'+page).classList.add('active');document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.page===page));const names={dashboard:'Overview',monitoring:'Live monitoring',members:'Members',logs:'Access logs',occupancy:'Inside now',unknown:'Unknown review',cameras:'Camera management',settings:'AI & operating settings'};$('pageTitle').textContent=names[page]||'Overview';if(page==='members')renderMembers();if(page==='logs')renderEvents();if(page==='unknown')renderUnknown();if(page==='occupancy')renderInside();if(page==='cameras')loadCameraSettings();window.scrollTo({top:0,behavior:'smooth'})}
document.querySelectorAll('.nav').forEach(b=>b.addEventListener('click',()=>showPage(b.dataset.page)));
async function api(url,opts={}){const r=await fetch(url,opts);let d={};try{d=await r.json()}catch{}if(!r.ok)throw new Error(d.error||'Request failed');return d}
async function loadStatus(){try{const d=await api('/api/status');$('metricMembers').textContent=d.members;$('metricInside').textContent=d.inside;$('metricEntries').textContent=d.entries_today;$('metricUnknown').textContent=d.unknown_today;$('occupancyTotal').textContent=d.inside;$('unknownBadge').textContent=d.unknown_today;$('sidebarStatus').textContent=d.status;updateCameraStatuses(d.cameras||{});$('healthAI').textContent=d.cameras&&Object.values(d.cameras).some(c=>c.status.includes('AI active'))?'Loaded / active':(d.cameras&&Object.values(d.cameras).some(c=>c.status.includes('AI issue'))?'AI issue':'Loads on first recognition');}catch(e){console.error(e)}}
function updateCameraStatuses(statuses){for(const role of ['entry','exit']){const s=statuses[role]||{};const up=!!s.running;const title=role[0].toUpperCase()+role.slice(1);const pillIds=role==='entry'?['dashEntryPill','monEntryPill']:['dashExitPill','monExitPill'];pillIds.forEach(id=>{if($(id)){$(id).className='state-pill '+(up?'online':'');$(id).innerHTML='<span class="tiny-dot"></span> '+(up?'LIVE':'OFFLINE')}});[['dash'+title+'Status',s.status||'Not connected'],['dash'+title+'Overlay',s.status||'Not connected'],['dash'+title+'Source',s.label||s.configured_type||'USB camera'],['mon'+title+'Label',s.name||title+' Camera'],['mon'+title+'Overlay',s.status||'Not connected'],['mon'+title+'Error',s.error||`${s.fps||0} FPS capture`],['health'+title,up?(s.status||'Connected'):'Not connected']].forEach(([id,v])=>{if($(id))$(id).textContent=v});}}
async function loadAll(){try{[members,events,inside]=await Promise.all([api('/api/members'),api('/api/events?limit=300'),api('/api/inside')]);renderMembers();renderEvents();renderInside();renderUnknown();renderRecent()}catch(e){toast(e.message,true)}await loadStatus()}
function renderMembers(){const q=($('memberSearch')?.value||'').toLowerCase();const list=members.filter(m=>(m.name+' '+m.member_code+' '+m.email).toLowerCase().includes(q));if($('memberCount'))$('memberCount').textContent=list.length+' members';$('membersTable').innerHTML=list.length?`<table class="data-table"><thead><tr><th>Member</th><th>Member code</th><th>Face template</th><th>Added</th><th>Actions</th></tr></thead><tbody>${list.map(m=>`<tr><td><span class="table-avatar">${esc((m.name||'?').slice(0,1).toUpperCase())}</span><strong>${esc(m.name)}</strong><br><small>${esc(m.email||'')}</small></td><td>${esc(m.member_code)}</td><td><span class="pill ${m.has_face?'inside':'outside'}">${(m.template_count>=3)?'3 ANGLES ('+m.template_count+')':(m.template_count>0?m.template_count+' ENROLLED':'NOT ENROLLED')}</span></td><td>${esc((m.created_at||'').slice(0,10))}</td><td><button class="action-link" onclick="manualEvent(${m.id},'ENTRY')">Entry</button><button class="action-link" onclick="manualEvent(${m.id},'EXIT')">Exit</button><button class="action-link" onclick="startEnroll(${m.id})">Enroll face</button><button class="action-link danger" onclick="deleteMember(${m.id})">Delete</button></td></tr>`).join('')}</tbody></table>`:'<div class="empty">No members found. Add a member to get started.</div>'}
function eventPill(e){return `<span class="pill ${e==='ENTRY'?'entry':e==='EXIT'?'exit':'unknown'}">${esc(e)}</span>`}
function renderEvents(){const q=($('eventSearch')?.value||'').toLowerCase(),f=$('eventFilter')?.value||'all';const list=events.filter(e=>(f==='all'||e.event_type===f)&&(e.member_name+' '+e.member_code+' '+e.event_type+' '+e.event_time+' '+e.source).toLowerCase().includes(q));if($('eventCount'))$('eventCount').textContent=list.length+' records';$('eventsTable').innerHTML=list.length?`<table class="data-table"><thead><tr><th>Time</th><th>Person</th><th>Code</th><th>Event</th><th>Confidence</th><th>Camera/source</th></tr></thead><tbody>${list.map(e=>`<tr><td>${esc(e.event_time.replace('T',' '))}</td><td><strong>${esc(e.member_name)}</strong></td><td>${esc(e.member_code||'—')}</td><td>${eventPill(e.event_type)}</td><td>${Number(e.confidence||0).toFixed(2)}</td><td>${esc(e.source||'—')}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No access events recorded yet.</div>'}
function renderInside(){const html=inside.length?`<table class="data-table"><thead><tr><th>Member</th><th>Code</th><th>Entry time</th><th>Status</th></tr></thead><tbody>${inside.map(m=>`<tr><td><strong>${esc(m.name)}</strong></td><td>${esc(m.member_code)}</td><td>${esc(m.entry_time.replace('T',' '))}</td><td><span class="pill inside">INSIDE</span></td></tr>`).join('')}</tbody></table>`:'<div class="empty">No members currently marked inside.</div>';if($('insideTable'))$('insideTable').innerHTML=html;if($('insidePreview'))$('insidePreview').innerHTML=html}
function renderUnknown(){const list=events.filter(e=>e.event_type==='UNKNOWN');$('unknownTable').innerHTML=list.length?`<table class="data-table"><thead><tr><th>Detected at</th><th>Detection</th><th>Source camera</th><th>Review</th></tr></thead><tbody>${list.map(e=>`<tr><td>${esc(e.event_time.replace('T',' '))}</td><td><strong>Unknown person</strong></td><td>${esc(e.source||'—')}</td><td>${e.reviewed?'<span class="pill inside">REVIEWED</span>':`<button class="action-link" onclick="reviewEvent(${e.id})">Mark reviewed</button>`}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">No unknown face events. Unknown events appear when AI is running and an unmatched face is seen.</div>'}
function renderRecent(){const list=events.slice(0,6);$('recentActivity').innerHTML=list.length?list.map(e=>`<div class="activity-item"><div class="event-icon ${e.event_type.toLowerCase()}">${e.event_type==='ENTRY'?'↘':e.event_type==='EXIT'?'↗':'⚑'}</div><div class="event-copy"><strong>${esc(e.member_name)}</strong><small>${esc(e.member_code||'No member match')} · ${esc(e.event_type)} · ${esc(e.source||'')}</small></div><span class="event-time">${esc((e.event_time||'').slice(11,16))}</span></div>`).join(''):'<div class="empty">No activity recorded yet.</div>'}
function openMemberModal(){editingMemberId=null;$('memberForm').reset();$('memberForm').classList.remove('hidden');$('memberModal').classList.remove('hidden');$('modalTitle').textContent='Add member';$('enrollArea').classList.add('hidden')}
function closeMemberModal(){$('memberModal').classList.add('hidden');$('memberForm').classList.remove('hidden')}
$('memberForm').addEventListener('submit',async e=>{e.preventDefault();try{const d=await api('/api/members',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({member_code:$('member_code').value,name:$('member_name').value,email:$('member_email').value,notes:$('member_notes').value})});editingMemberId=d.id;$('modalTitle').textContent='Member saved';$('enrollArea').classList.remove('hidden');$('memberForm').classList.add('hidden');toast('Member saved. Enroll a clear face photo next.');await loadAll()}catch(err){toast(err.message,true)}});
function startEnroll(id){
  editingMemberId=id;
  $('memberForm').reset();
  ['Front','Left','Right'].forEach(slot=>{
    const inp=$('photo'+slot);if(inp)inp.value='';
    const card=$('slotCard'+slot);if(card)card.classList.remove('has-file');
    const prev=$('preview'+slot);if(prev)prev.innerHTML=`<span>👤 ${slot==='Front'?'Straight':'Turn '+slot}</span>`;
    const badge=$('badge'+slot);if(badge)badge.textContent=(slot==='Front'?'Required':'~30° Turn');
  });
  $('memberModal').classList.remove('hidden');
  $('modalTitle').textContent='Enroll member face';
  $('enrollArea').classList.remove('hidden');
  $('memberForm').classList.add('hidden');
}

let enrollMultiMode = true;
function toggleEnrollMode(){
  enrollMultiMode = !enrollMultiMode;
  $('multiEnrollWrap').classList.toggle('hidden', !enrollMultiMode);
  $('singleEnrollWrap').classList.toggle('hidden', enrollMultiMode);
  $('btnToggleEnrollMode').textContent = enrollMultiMode ? 'Single Photo Mode' : '3-Angle Mode';
}

function handleEnrollPreview(slot){
  const inp = $('photo' + slot);
  const prev = $('preview' + slot);
  const card = $('slotCard' + slot);
  const badge = $('badge' + slot);
  if(inp && inp.files && inp.files[0]){
    const file = inp.files[0];
    const reader = new FileReader();
    reader.onload = e => {
      prev.innerHTML = `<img src="${e.target.result}" alt="${slot}">`;
      card.classList.add('has-file');
      if(badge) badge.textContent = 'Selected';
    };
    reader.readAsDataURL(file);
  }
}

async function enrollMulti(){
  if(!editingMemberId) return toast('Save a member first.', true);
  const fFront = $('photoFront')?.files[0];
  const fLeft = $('photoLeft')?.files[0];
  const fRight = $('photoRight')?.files[0];
  if(!fFront && !fLeft && !fRight){
    return toast('Please select at least 1 photo (Front view recommended).', true);
  }
  const fd = new FormData();
  if(fFront) fd.append('photo_front', fFront);
  if(fLeft) fd.append('photo_left', fLeft);
  if(fRight) fd.append('photo_right', fRight);

  const btn = $('btnEnrollMulti');
  if(btn){ btn.disabled = true; btn.textContent = 'Enrolling 3 angles...'; }
  try {
    const d = await api(`/api/members/${editingMemberId}/enroll-multi`, { method: 'POST', body: fd });
    toast(d.message);
    if(d.warnings && d.warnings.length) toast(d.warnings.join('; '), true);
    closeMemberModal();
    await loadAll();
  } catch(e) {
    toast(e.message, true);
  } finally {
    if(btn){ btn.disabled = false; btn.textContent = 'Enroll 3-Angle Profile'; }
  }
}

async function enrollFace(){if(!editingMemberId)return toast('Save a member first.',true);const f=$('facePhoto').files[0];if(!f)return toast('Choose a face photo first.',true);const fd=new FormData();fd.append('photo',f);try{const d=await api(`/api/members/${editingMemberId}/enroll-face`,{method:'POST',body:fd});toast(d.message);closeMemberModal();await loadAll()}catch(e){toast(e.message,true)}}
async function deleteMember(id){if(!confirm('Delete this member and their visits? Historical event records are retained.'))return;try{await api('/api/members/'+id,{method:'DELETE'});toast('Member deleted');await loadAll()}catch(e){toast(e.message,true)}}
async function manualEvent(id,type){try{await api('/api/manual-event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({member_id:id,event_type:type})});toast(type+' recorded');await loadAll()}catch(e){toast(e.message,true)}}
async function reviewEvent(id){try{await api('/api/events/'+id+'/review',{method:'POST'});toast('Marked as reviewed');await loadAll()}catch(e){toast(e.message,true)}}
async function cameraAction(role,action){try{const d=await api(`/api/cameras/${role}/${action}`,{method:'POST'});toast(d.message||`${role} camera ${action==='start'?'connected':'disconnected'}`);await loadStatus()}catch(e){toast(e.message,true);await loadStatus()}}
async function startAllCameras(){for(const role of ['entry','exit']){try{await api(`/api/cameras/${role}/start`,{method:'POST'})}catch(e){toast(`${role}: ${e.message}`,true)}}await loadStatus()}
async function stopAllCameras(){await Promise.all(['entry','exit'].map(role=>cameraAction(role,'stop')))}
async function loadCameraSettings(){try{const d=await api('/api/cameras');cameraConfig=d.config||{};for(const role of ['entry','exit']){const c=cameraConfig[role]||{};$(`${role}_name`).value=c.name||`${role} Camera`;$(`${role}_type`).value=c.camera_type||'usb';$(`${role}_enabled`).checked=c.enabled!==false;if(c.camera_type==='rtsp')$(`${role}_source_rtsp`).value=c.source||'';else window[`selectedUsb_${role}`]=String(c.source??(role==='entry'?'0':'1'));updateCameraInput(role)}await populateDeviceOptions();await renderDetectedDevices()}catch(e){console.error(e)}}
function updateCameraInput(role){const type=$(`${role}_type`).value;$(`${role}_usb_wrap`).classList.toggle('hidden',type!=='usb');$(`${role}_rtsp_wrap`).classList.toggle('hidden',type!=='rtsp')}
async function populateDeviceOptions(){for(const role of ['entry','exit']){const select=$(`${role}_source_usb`);const selected=String(window[`selectedUsb_${role}`]??(role==='entry'?'0':'1'));let opts=discovered.map(d=>`<option value="${esc(d.source)}">${esc(d.label)}</option>`);for(const idx of ['0','1','2','3','4','5'])if(!discovered.some(d=>String(d.source)===idx))opts.push(`<option value="${idx}">USB camera ${idx} (not detected in last scan)</option>`);select.innerHTML=opts.join('');select.value=selected;if(!select.value)select.value=selected;}}
async function discoverCameras(force=false){$('discoveryStatus').textContent='Scanning camera indexes 0–5… this can take a few seconds.';try{const d=await api('/api/cameras/discover?refresh='+(force?'1':'0'));discovered=d.devices||[];await populateDeviceOptions();await renderDetectedDevices();$('discoveryStatus').textContent=d.message+' Use Test source to confirm each camera.';toast(d.message)}catch(e){$('discoveryStatus').textContent='Camera scan failed: '+e.message;toast(e.message,true)}}
async function renderDetectedDevices(){if(!discovered.length){$('discoveredTable').innerHTML='<div class="empty">No available camera detected yet. Try refresh, check cable and Windows camera privacy permissions.</div>';return}$('discoveredTable').innerHTML=`<table class="data-table"><thead><tr><th>Device index</th><th>OpenCV label</th><th>Detected</th><th>Assignment</th></tr></thead><tbody>${discovered.map(d=>{const inUse=['entry','exit'].filter(role=>cameraConfig[role]&&cameraConfig[role].camera_type==='usb'&&String(cameraConfig[role].source)===String(d.source)).map(r=>r.toUpperCase()).join(', ');return `<tr><td><strong>${esc(d.source)}</strong></td><td>${esc(d.label)}</td><td><span class="pill inside">AVAILABLE</span></td><td>${esc(inUse||'Not assigned')}</td></tr>`}).join('')}</tbody></table>`}
function cameraFormPayload(role){const type=$(`${role}_type`).value;return{name:$(`${role}_name`).value,camera_type:type,source:type==='usb'?$(`${role}_source_usb`).value:$(`${role}_source_rtsp`).value,enabled:$(`${role}_enabled`).checked}}
async function saveCamera(event,role){event.preventDefault();const payload=cameraFormPayload(role);try{await api(`/api/cameras/${role}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});cameraConfig[role]=payload;toast(`${role[0].toUpperCase()+role.slice(1)} camera settings saved`);await loadStatus();await renderDetectedDevices()}catch(e){toast(e.message,true)}}
async function testCamera(role){const payload=cameraFormPayload(role);try{await api(`/api/cameras/${role}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});cameraConfig[role]=payload;const d=await api(`/api/cameras/${role}/test`,{method:'POST'});$(`${role}_result`).className='test-result success';$(`${role}_result`).textContent=d.message;toast(d.message)}catch(e){$(`${role}_result`).className='test-result failure';$(`${role}_result`).textContent=e.message;toast(e.message,true)}}
async function loadSettings(){try{config=await api('/api/config');$('recognition_threshold').value=config.recognition_threshold??0.42;$('recognition_interval').value=config.recognition_interval??0.4;$('operating_start').value=config.operating_start??'08:30';$('operating_end').value=config.operating_end??'16:30';$('unknown_alerts').checked=!!config.unknown_alerts;const h=await api('/api/health');$('healthProvider').textContent=h.face_model_loaded?'Loaded':(h.face_model_error?'Model not loaded':'Loads on first use')}catch(e){console.error(e)}}
$('settingsForm').addEventListener('submit',async e=>{e.preventDefault();try{await api('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({recognition_threshold:Number($('recognition_threshold').value),recognition_interval:Number($('recognition_interval').value),operating_start:$('operating_start').value,operating_end:$('operating_end').value,unknown_alerts:$('unknown_alerts').checked})});toast('AI settings saved');await loadSettings()}catch(e){toast(e.message,true)}});
$('dateLabel').textContent='LOCAL WORKSPACE / '+new Date().toLocaleDateString(undefined,{weekday:'short',day:'2-digit',month:'short',year:'numeric'}).toUpperCase();
activateVideoFeeds('dashboard');loadSettings().then(async()=>{await loadCameraSettings();await loadAll();await discoverCameras(false)});setInterval(loadStatus,2000);setInterval(loadAll,10000);
