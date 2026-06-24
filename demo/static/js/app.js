
const generateBtn = document.getElementById("generateBtn");

const keywords = document.getElementById("keywords");

const headline = document.getElementById("headline");

const loading = document.getElementById("loading");

const copyBtn = document.getElementById("copyBtn");

const clearBtn = document.getElementById("clearBtn");

const historyList = document.getElementById("history");

const totalGenerate = document.getElementById("count");

const inferTime = document.getElementById("time")


// ========================================
// Load history
// ========================================

let history = JSON.parse(
    localStorage.getItem("headline_history") || "[]"
);

renderHistory();


// ========================================
// Submit
// ========================================

generateBtn.addEventListener("click", async () => {
    // XÓA dòng e.preventDefault(); vì nút bấm thường không cần cái này

    const text = keywords.value.trim();
    if(text===""){
        alert("Nhập từ khóa trước nhé!");
        return;
    }

    loading.classList.remove("hidden");

    headline.innerHTML="";

    const start = performance.now();

    try{

        const response = await fetch("/predict",{

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify({

                keywords:text

            })

        });

        const data = await response.json();

        const end = performance.now();

        headline.innerHTML = data.headline;

        inferTime.innerHTML =
            (end-start).toFixed(0)+" ms";

        history.unshift({

            keyword:text,

            headline:data.headline

        });

        if(history.length>10){

            history.pop();

        }

        localStorage.setItem(
            "headline_history",
            JSON.stringify(history)
        );

        renderHistory();

        updateCounter();

    }

    catch(error){

        headline.innerHTML="Có lỗi xảy ra.";

    }

    loading.classList.add("hidden");

});


// ========================================
// Copy
// ========================================

copyBtn.addEventListener("click",()=>{

    navigator.clipboard.writeText(

        headline.innerText

    );

    showToast("Đã copy!");

});


// ========================================
// Clear
// ========================================

clearBtn.addEventListener("click",()=>{

    keywords.value="";

    headline.innerHTML="";

});


// ========================================
// Ctrl + Enter
// ========================================

keywords.addEventListener("keydown",(e)=>{

    if(e.ctrlKey && e.key==="Enter"){

        form.requestSubmit();

    }

});


// ========================================
// Counter
// ========================================

function updateCounter(){

    let total = Number(

        localStorage.getItem("total_generate") || 0

    );

    total++;

    localStorage.setItem(

        "total_generate",

        total

    );

    totalGenerate.innerHTML=total;

}

totalGenerate.innerHTML=

localStorage.getItem("total_generate")||0;


// ========================================
// History
// ========================================

function renderHistory(){

    historyList.innerHTML="";

    history.forEach(item=>{

        const li=document.createElement("li");

        li.innerHTML=

        "<b>"+item.keyword+"</b><br>"+

        item.headline;

        li.onclick=()=>{

            keywords.value=item.keyword;

            headline.innerHTML=item.headline;

        };

        historyList.appendChild(li);

    });

}


// ========================================
// Toast
// ========================================

function showToast(message){

    const toast=document.getElementById("toast");

    toast.innerHTML=message;

    toast.classList.add("show");

    setTimeout(()=>{

        toast.classList.remove("show");

    },2000);

}

